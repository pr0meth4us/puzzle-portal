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
    overlay_path = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference/scratch/overlay_sift.png")
    with open(overlay_path, 'rb') as f:
        data = f.read()
    
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "This is an overlay image of two line drawings after feature-based alignment. "
            "Look closely at the lines of the drawings (e.g. elephant, trees, crocodiles). "
            "Are the lines sharp and perfectly overlapping, or are they ghosted/blurry, indicating that the alignment failed?",
            types.Part.from_bytes(data=data, mime_type="image/png")
        ]
    )
    print("Gemini response for overlay_sift.png:")
    print(response.text)

if __name__ == '__main__':
    asyncio.run(main())
