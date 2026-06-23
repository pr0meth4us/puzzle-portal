import cv2
import sys
import asyncio
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.bifrost_config import get_config

load_dotenv(Path(__file__).resolve().parents[2] / '.env')
GEMINI_API_KEY = get_config('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)

async def main():
    p1_path = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference/validation_dataset/puzzle_01.png")
    
    with open(p1_path, 'rb') as f: p1_data = f.read()
    
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "Please describe the content and layout of this image. "
            "Is it a line drawing of animals (elephant, crocodile, etc.)? "
            "Is it a combined image with two panels?",
            types.Part.from_bytes(data=p1_data, mime_type="image/png")
        ]
    )
    print("Gemini response for puzzle_01.png:")
    print(response.text)

if __name__ == '__main__':
    asyncio.run(main())
