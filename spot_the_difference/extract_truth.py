import os
import sys
import asyncio
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from utils.bifrost_config import get_config

load_dotenv(PROJECT_DIR / '.env')
GEMINI_API_KEY = get_config('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)

async def main():
    val_dir = SCRIPT_DIR / 'validation_dataset'
    answers = sorted(val_dir.glob('answer_*.jpg'))
    for ans in answers:
        with open(ans, 'rb') as f: data = f.read()
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "This is an answer key for a 'spot the difference' puzzle. How many differences are circled or indicated? Respond with ONLY a single integer.",
                types.Part.from_bytes(data=data, mime_type="image/jpeg")
            ]
        )
        print(f"{ans.name}: {response.text.strip()}")

if __name__ == '__main__':
    asyncio.run(main())
