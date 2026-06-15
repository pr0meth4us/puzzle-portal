#!/usr/bin/env python3
import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from google import genai
from google.genai import types

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
load_dotenv(PROJECT_DIR / '.env')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

PROMPT = """You are a master puzzle router. Look at the image provided and classify it into EXACTLY ONE of the following categories:

- "SPOT_DIFFERENCE": The image contains two nearly identical scenes (usually top/bottom or left/right) and the user has to find differences.
- "NUMBER_GRID": The image contains a grid of numbers where a specific sequence needs to be found.
- "VISUAL_PUZZLE": The image is a visual riddle, an optical illusion, or asks to count animals/objects (e.g., hidden cats, faces).

Respond with ONLY a JSON object:
{
  "category": "SPOT_DIFFERENCE" | "NUMBER_GRID" | "VISUAL_PUZZLE",
  "reasoning": "brief explanation"
}
"""

async def classify_puzzle(image_path: Path):
    client = genai.Client(api_key=GEMINI_API_KEY)
    with open(image_path, 'rb') as f:
        data = f.read()
    
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    config = types.GenerateContentConfig(response_mime_type="application/json")
    
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            PROMPT,
            types.Part.from_bytes(data=data, mime_type=mime_type)
        ],
        config=config
    )
    
    try:
        return json.loads(response.text.strip())
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {"category": "VISUAL_PUZZLE", "reasoning": "fallback"}

def route_to_worker(image_path: Path, category: str):
    python_bin = PROJECT_DIR / 'venv' / 'bin' / 'python'
    
    if category == "SPOT_DIFFERENCE":
        script = PROJECT_DIR / 'spot_the_difference' / 'spot_the_differences.py'
        print(f"🚀 Routing to Spot the Difference Worker...")
        subprocess.run([str(python_bin), str(script), str(image_path)])
        
    elif category == "NUMBER_GRID":
        script = PROJECT_DIR / 'spot_the_difference' / 'spot_the_differences.py'
        print(f"🚀 Routing to Number Grid Worker...")
        subprocess.run([str(python_bin), str(script), str(image_path), "--mode", "number"])
        
    elif category == "VISUAL_PUZZLE":
        script = PROJECT_DIR / 'ocr_tools' / 'visual_solver.py'
        print(f"🚀 Routing to General Visual Puzzle Worker...")
        subprocess.run([str(python_bin), str(script), str(image_path)])
        
    else:
        print(f"❌ Unknown category: {category}")

async def main():
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found in .env")
        sys.exit(1)
        
    if len(sys.argv) < 2:
        print("Usage: python router.py <path_to_puzzle_image>")
        sys.exit(1)
        
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"ERROR: Image not found at {image_path}")
        sys.exit(1)
        
    print(f"🔍 Analyzing puzzle type: {image_path.name}...")
    classification = await classify_puzzle(image_path)
    
    print(f"🎯 Classification: {classification.get('category')} ({classification.get('reasoning')})")
    print("-" * 50)
    route_to_worker(image_path, classification.get('category'))

if __name__ == "__main__":
    asyncio.run(main())
