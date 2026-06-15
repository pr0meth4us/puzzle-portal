#!/usr/bin/env python3
import os
import sys
import time
import json
import asyncio
import subprocess
import platform
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from google.cloud import vision
from google import genai
from google.genai import types
from utils.bifrost_config import get_config

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

load_dotenv(dotenv_path=PROJECT_DIR / '.env')

GOOGLE_APPLICATION_CREDENTIALS = get_config(
    'GOOGLE_APPLICATION_CREDENTIALS',
    str(PROJECT_DIR / 'credentials.json')
)
GEMINI_API_KEY = get_config('GEMINI_API_KEY')

IMAGE_FOLDER = SCRIPT_DIR / 'khmer_test_images'
OUTPUT_FOLDER = SCRIPT_DIR / 'results'
OUTPUT_FOLDER.mkdir(exist_ok=True)

PROMPT = """You are a highly intelligent puzzle-solving assistant. You will receive an image and its extracted Khmer text via OCR.
Look at the image carefully to solve any visual puzzles, count objects (like cats or hidden numbers), spot optical illusions, or play find-the-item games.
Read the text to understand the specific question.
Respond with ONLY a JSON object containing these keys:
- "final_answer": The direct, final answer to the question in Khmer (concise, no emojis).
- "explanation": A detailed explanation or reasoning for the answer in Khmer.
- "explicitly_asks_to_explain": Boolean (true/false), whether the input text explicitly asks you to explain the answer.
- "asks_to_draw_or_visualize": Boolean (true/false), whether the input text asks to draw, visualize, graph, chart something, or tap on the image to show answers (e.g., "ចុចបង្ហាញ").
Make sure to return valid JSON."""

def copy_to_clipboard(text):
    current_os = platform.system()
    try:
        if current_os == 'Windows':
            subprocess.run(['clip'], input=text.encode('utf-16'), check=True)
        elif current_os == 'Darwin':
            subprocess.run(['pbcopy'], input=text.encode('utf-8'), check=True)
        elif current_os == 'Linux':
            try:
                subprocess.run(['xclip', '-selection', 'clipboard'], input=text.encode('utf-8'), check=True)
            except FileNotFoundError:
                subprocess.run(['xsel', '--clipboard', '--input'], input=text.encode('utf-8'), check=True)
        else:
            print(f"  [clipboard not supported on OS: {current_os}]")
    except Exception as e:
        print(f"  [clipboard failed: {e}]")

def setup():
    os.environ['GRPC_ENABLE_FORK_SUPPORT'] = 'False'
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS
    vision_client = vision.ImageAnnotatorClient()
    return vision_client, genai.Client(api_key=GEMINI_API_KEY)

def get_newest_image():
    images = (
        list(IMAGE_FOLDER.glob('*.png')) +
        list(IMAGE_FOLDER.glob('*.jpg')) +
        list(IMAGE_FOLDER.glob('*.jpeg'))
    )
    if not images:
        return None
    return max(images, key=lambda f: f.stat().st_mtime)

async def process(image_path, vision_client, genai_client):
    pipeline_start = time.time()

    print(f"OCR: Processing {image_path.name} with Cloud Vision for highest accuracy...")
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        image = vision.Image(content=image_bytes)
        
    response = vision_client.document_text_detection(image=image)
    extracted_text = response.full_text_annotation.text
    ocr_time = time.time() - pipeline_start

    print(f"LLM: Sending text and image to Gemini 2.5 Flash asynchronously...")
    answer_start = time.time()
    
    config = types.GenerateContentConfig(response_mime_type="application/json")
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    
    answer_response = await genai_client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            f"{PROMPT}\n\nOCR Text:\n{extracted_text}",
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        ],
        config=config
    )
    
    try:
        parsed_answer = json.loads(answer_response.text.strip())
    except Exception as e:
        print(f"Warning: Failed to parse JSON response: {e}")
        parsed_answer = {
            "final_answer": answer_response.text.strip(),
            "explanation": "Could not parse explanation.",
            "explicitly_asks_to_explain": False,
            "asks_to_draw_or_visualize": False
        }

    answer_time = time.time() - answer_start
    total_time = time.time() - pipeline_start

    return {
        'image': image_path.name,
        'extracted_text': extracted_text,
        'parsed_answer': parsed_answer,
        'timing': {
            'ocr_seconds': round(ocr_time, 3),
            'llm_seconds': round(answer_time, 3),
            'total_seconds': round(total_time, 3)
        },
        'status': 'SUCCESS',
        'timestamp': datetime.now().isoformat()
    }

async def main_async():
    if not Path(GOOGLE_APPLICATION_CREDENTIALS).exists():
        print(f"ERROR: {GOOGLE_APPLICATION_CREDENTIALS} not found")
        sys.exit(1)
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not in .env")
        sys.exit(1)
    if not IMAGE_FOLDER.exists():
        print(f"ERROR: {IMAGE_FOLDER} folder not found")
        sys.exit(1)

    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        image_path = get_newest_image()

    if not image_path or not image_path.exists():
        print(f"ERROR: No images found or invalid path")
        sys.exit(1)

    print(f"Processing image: {image_path.name}")

    vision_client, genai_client = setup()

    try:
        result = await process(image_path, vision_client, genai_client)

        print("\n--- Extracted Text (Question) ---")
        print(result['extracted_text'].strip() if result['extracted_text'] else "(No text detected)")
        print("---------------------------------")

        parsed = result['parsed_answer']
        
        if parsed.get('asks_to_draw_or_visualize'):
            print("\n\033[41;97;1;5m 🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨 \033[0m")
            print("\033[41;97;1;5m 🚨 ALERT: THE IMAGE PROMPT ASKS TO DRAW OR VISUALIZE! 🚨 \033[0m")
            print("\033[41;97;1;5m 🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨 \033[0m\n")

        print(f"\nExplanation:\n{parsed.get('explanation')}")
        print(f"\nFinal Answer: {parsed.get('final_answer')}")

        clipboard_text = parsed.get('final_answer')
        if parsed.get('explicitly_asks_to_explain'):
            clipboard_text = f"{parsed.get('final_answer')}\n\nExplanation:\n{parsed.get('explanation')}"

        print(f"\nTime: {result['timing']['total_seconds']}s (OCR: {result['timing']['ocr_seconds']}s | LLM: {result['timing']['llm_seconds']}s)")

        copy_to_clipboard(clipboard_text)
        print("Copied to clipboard.")

        output_file = OUTPUT_FOLDER / f"visual_solve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main_async())
