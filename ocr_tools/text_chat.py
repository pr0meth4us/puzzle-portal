#!/usr/bin/env python3
import os
import sys
import time
import json
import asyncio
import platform
import subprocess
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from utils.bifrost_config import get_config
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = 'False'

from google import genai
from google.genai import types

GOOGLE_APPLICATION_CREDENTIALS = get_config(
    'GOOGLE_APPLICATION_CREDENTIALS',
    str(PROJECT_DIR / 'credentials.json')
)
GEMINI_API_KEY = get_config('GEMINI_API_KEY')

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

def setup():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS
    vertex_project = os.getenv("VERTEX_AI_PROJECT", "khmer-ocr-496606")
    vertex_location = os.getenv("VERTEX_AI_LOCATION", "asia-southeast1")
    return genai.Client(
        vertexai=True, 
        project=vertex_project, 
        location=vertex_location
    )

async def process(text_input, genai_client):
    print(f"LLM: Sending text to Gemini 3.5 Flash asynchronously...")
    answer_start = time.time()
    
    config = types.GenerateContentConfig(response_mime_type="application/json")
    answer_response = await genai_client.aio.models.generate_content(
        model="gemini-3.5-flash",
        contents=f"{PROMPT}\n\n{text_input}",
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

    return {
        'input_text': text_input,
        'parsed_answer': parsed_answer,
        'timing': {
            'llm_seconds': round(answer_time, 3),
        },
        'status': 'SUCCESS'
    }

async def main_async():
    if not Path(GOOGLE_APPLICATION_CREDENTIALS).exists():
        print(f"ERROR: {GOOGLE_APPLICATION_CREDENTIALS} not found")
        sys.exit(1)
        
    genai_client = setup()
    
    if len(sys.argv) > 1:
        # Command line argument mode
        text_input = " ".join(sys.argv[1:])
    else:
        # Read from standard input if piped
        if not sys.stdin.isatty():
            text_input = sys.stdin.read().strip()
        else:
            print("ERROR: Please provide text as an argument or pipe it via stdin.")
            sys.exit(1)
            
    if not text_input:
        print("ERROR: No input text provided.")
        sys.exit(1)
        
    result = await process(text_input, genai_client)
    print_result(result)

def print_result(result):
    parsed = result['parsed_answer']
    
    if parsed.get('asks_to_draw_or_visualize'):
        print("\n\033[41;97;1;5m 🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨 \033[0m")
        print("\033[41;97;1;5m 🚨 ALERT: THE PROMPT ASKS TO DRAW OR VISUALIZE! 🚨 \033[0m")
        print("\033[41;97;1;5m 🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨 \033[0m\n")

    if parsed.get('reasoning'):
        print(f"\n🧠 [REASONING]:\n{parsed.get('reasoning')}")

    if parsed.get('explicitly_asks_to_explain'):
        print(f"\nExplanation:\n{parsed.get('explanation')}")
        
    print(f"\nFinal Answer: {parsed.get('final_answer')}")
    print(f"\nTime: {result['timing']['llm_seconds']}s")
    
    clipboard_text = parsed.get('final_answer')
    if parsed.get('explicitly_asks_to_explain'):
        clipboard_text = f"{parsed.get('final_answer')}\n\nExplanation:\n{parsed.get('explanation')}"
        
    copy_to_clipboard(clipboard_text)
    print("Copied to clipboard.")

if __name__ == '__main__':
    asyncio.run(main_async())
