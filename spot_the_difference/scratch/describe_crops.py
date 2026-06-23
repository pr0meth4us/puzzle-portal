import os
import sys
import asyncio
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.bifrost_config import get_config

load_dotenv(Path(__file__).resolve().parents[2] / '.env')
GEMINI_API_KEY = get_config('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)

async def describe_crop(img_path):
    with open(img_path, 'rb') as f:
        data = f.read()
    
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "This is a crop from a spot-the-difference puzzle. "
            "The left half is from the first image, and the right half is from the second image. "
            "Please describe what is shown (e.g. text, cloud, bird, spider, etc.) and state clearly "
            "if there is a difference between the left and right halves.",
            types.Part.from_bytes(data=data, mime_type="image/png")
        ]
    )
    return response.text

async def main():
    crops_dir = Path("scratch/crops")
    files = sorted([f for f in os.listdir(crops_dir) if f.endswith(".png")])
    
    for f in files:
        path = crops_dir / f
        desc = await describe_crop(path)
        print(f"\n=== {f} ===")
        print(desc.strip())

if __name__ == '__main__':
    asyncio.run(main())
