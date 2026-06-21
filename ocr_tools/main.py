#!/Users/nicksng/code/random/venv/bin/python
import os
import sys
import time
import json
import asyncio
import subprocess
import platform
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from utils.bifrost_config import get_config
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = 'False'

from google.cloud import vision
from google import genai
from google.genai import types

_g_creds_raw = get_config(
    'GOOGLE_APPLICATION_CREDENTIALS',
    str(PROJECT_DIR / 'credentials.json')
)
_g_creds_path = Path(_g_creds_raw)
if not _g_creds_path.is_absolute():
    _g_creds_path = (PROJECT_DIR / _g_creds_path).resolve()
GOOGLE_APPLICATION_CREDENTIALS = str(_g_creds_path)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS

GEMINI_API_KEY = get_config('GEMINI_API_KEY')

IMAGE_FOLDER = SCRIPT_DIR / 'khmer_test_images'
OUTPUT_FOLDER = SCRIPT_DIR / 'results'
OUTPUT_FOLDER.mkdir(exist_ok=True)

PROMPT = """You are a helpful assistant reading Khmer text. Read it carefully and think through the question or puzzle.
Respond with ONLY a JSON object containing these keys:
- "reasoning": Think step-by-step in Khmer about the riddle or question before providing the final answer. What is the text asking? What clues are given?
- "final_answer": The direct, final answer to the question in Khmer (concise, no emojis).
- "explicitly_asks_to_explain": Boolean (true/false), whether the input text explicitly asks you to explain the answer.
- "explanation": A detailed explanation or reasoning for the answer in Khmer. ONLY provide this if explicitly_asks_to_explain is true. Otherwise, leave it as an empty string.
- "asks_to_draw_or_visualize": Boolean (true/false), whether the input text asks to draw, visualize, graph, or chart something.
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

def sync_to_icloud(text):
    paths = [
        Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/clip.txt",
        Path.home() / "Library/Mobile Documents/iCloud~is~workflow~my~workflows/Documents/clip.txt"
    ]
    for p in paths:
        try:
            if p.parent.exists():
                p.write_text(text, encoding="utf-8")
                print(f"Synced to iCloud: {p.name} ({p.parent.name})")
        except Exception as e:
            print(f"  [iCloud sync failed for {p.parent.name}: {e}]")

def sync_to_local_server(text):
    try:
        local_file = Path("/tmp/local_clip.txt")
        local_file.write_text(text, encoding="utf-8")
        
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.2)
        result = s.connect_ex(('127.0.0.1', 8089))
        s.close()
        
        if result != 0:
            server_script = SCRIPT_DIR / "clip_server.py"
            subprocess.Popen([sys.executable, str(server_script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Local clip server started on port 8089.")
        else:
            print("Local clip server updated.")
    except Exception as e:
        print(f"  [Local server update failed: {e}]")

def clean_extracted_text(text: str) -> str:
    # Remove "រង្វាន់" and everything after it
    parts = re.split(r'រ\s*ង\s*្\s*វ\s*ា\s*ន\s*់?', text, flags=re.IGNORECASE)
    if len(parts) > 1:
        text = parts[0]
    text = text.strip()
    # Strip any trailing separator lines
    lines = text.splitlines()
    while lines and re.match(r'^[-\s._*=]+$', lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()

def setup_genai():
    # Initialize the Vertex AI GenAI Client - project loaded from Bifrost via .env
    vertex_project = os.getenv("VERTEX_AI_PROJECT", "khmer-ocr-496606")
    vertex_location = os.getenv("VERTEX_AI_LOCATION", "asia-southeast1")
    return genai.Client(
        vertexai=True, 
        project=vertex_project, 
        location=vertex_location
    )

def setup_vision():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS
    return vision.ImageAnnotatorClient()

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
        image = vision.Image(content=f.read())
        
    # Running Cloud Vision (Synchronous step as requested for accuracy)
    response = vision_client.document_text_detection(image=image)
    extracted_text = response.full_text_annotation.text
    ocr_time = time.time() - pipeline_start

    # Clean the text to remove prize lists and similar metadata
    cleaned_text = clean_extracted_text(extracted_text)

    print(f"LLM: Sending text to Gemini 2.5 Flash asynchronously...")
    answer_start = time.time()
    
    # Using client.aio for high-performance non-blocking async calling
    config = types.GenerateContentConfig(response_mime_type="application/json")
    answer_response = await genai_client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{PROMPT}\n\n{cleaned_text}",
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
        'extracted_text': cleaned_text,
        'parsed_answer': parsed_answer,
        'timing': {
            'ocr_seconds': round(ocr_time, 3),
            'llm_seconds': round(answer_time, 3),
            'total_seconds': round(total_time, 3)
        },
        'status': 'SUCCESS',
        'timestamp': datetime.now().isoformat()
    }

async def process_text(input_text, genai_client):
    pipeline_start = time.time()

    cleaned_text = clean_extracted_text(input_text)

    print(f"LLM: Sending text directly to Gemini 2.5 Flash asynchronously...")
    answer_start = time.time()
    
    # Using client.aio for high-performance non-blocking async calling
    config = types.GenerateContentConfig(response_mime_type="application/json")
    answer_response = await genai_client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{PROMPT}\n\n{cleaned_text}",
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
        'image': 'TEXT_INPUT',
        'extracted_text': cleaned_text,
        'parsed_answer': parsed_answer,
        'timing': {
            'ocr_seconds': 0.0,
            'llm_seconds': round(answer_time, 3),
            'total_seconds': round(total_time, 3)
        },
        'status': 'SUCCESS',
        'timestamp': datetime.now().isoformat()
    }

async def main_async():
    # If the user passes text directly via command-line arguments
    text_input = None
    if len(sys.argv) > 1:
        text_input = " ".join(sys.argv[1:]).lstrip('-')
        # Clean any surrounding quotes that the shell might pass
        if (text_input.startswith('"') and text_input.endswith('"')) or \
           (text_input.startswith("'") and text_input.endswith("'")):
            text_input = text_input[1:-1]
        text_input = text_input.strip()

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not in .env")
        sys.exit(1)

    if not text_input:
        if not Path(GOOGLE_APPLICATION_CREDENTIALS).exists():
            print(f"ERROR: {GOOGLE_APPLICATION_CREDENTIALS} not found")
            sys.exit(1)
        if not IMAGE_FOLDER.exists():
            print(f"ERROR: {IMAGE_FOLDER} folder not found")
            sys.exit(1)

        image_path = get_newest_image()
        if not image_path:
            print(f"ERROR: No images found in {IMAGE_FOLDER}")
            sys.exit(1)

        print(f"Processing newest image: {image_path.name}")

    genai_client = setup_genai()

    try:
        if text_input:
            result = await process_text(text_input, genai_client)
        else:
            vision_client = setup_vision()
            result = await process(image_path, vision_client, genai_client)

        # --- NEW CODE: Print the extracted question ---
        print("\n--- Extracted Text (Question) ---")
        print(result['extracted_text'].strip())
        print("---------------------------------")

        parsed = result['parsed_answer']
        
        if parsed.get('asks_to_draw_or_visualize'):
            print("\n\033[41;97;1;5m 🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨 \033[0m")
            print("\033[41;97;1;5m 🚨 ALERT: THE IMAGE PROMPT ASKS TO DRAW OR VISUALIZE! 🚨 \033[0m")
            print("\033[41;97;1;5m 🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨 \033[0m\n")

        if parsed.get('reasoning'):
            print(f"\n🧠 [REASONING]:\n{parsed.get('reasoning')}")

        if parsed.get('explicitly_asks_to_explain'):
            print(f"\nExplanation:\n{parsed.get('explanation')}")
        print(f"\nFinal Answer: {parsed.get('final_answer')}")

        clipboard_text = parsed.get('final_answer')
        if parsed.get('explicitly_asks_to_explain'):
            clipboard_text = f"{parsed.get('final_answer')}\n\nExplanation:\n{parsed.get('explanation')}"

        if result['timing']['ocr_seconds'] > 0:
            print(f"\nTime: {result['timing']['total_seconds']}s (OCR: {result['timing']['ocr_seconds']}s | LLM: {result['timing']['llm_seconds']}s)")
        else:
            print(f"\nTime: {result['timing']['total_seconds']}s (LLM: {result['timing']['llm_seconds']}s)")

        copy_to_clipboard(clipboard_text)
        
        # Audible alert
        if platform.system() == 'Darwin':
            try:
                subprocess.Popen(['afplay', '/System/Library/Sounds/Glass.aiff'])
            except Exception:
                pass
                
        # Big visual alert
        print("\n\033[42;97;1m ✨✨ SUCCESS: COPIED & SYNCED! ✨✨ \033[0m\n")

        sync_to_icloud(clipboard_text)
        sync_to_local_server(clipboard_text)

        output_file = OUTPUT_FOLDER / f"production_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # Launch the async event loop
    asyncio.run(main_async())