import cv2
import numpy as np
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
    ans_path = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference/correct_answers/answer_01.jpg")
    with open(ans_path, 'rb') as f:
        data = f.read()
    
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "This is an answer key image for a spot the difference puzzle. "
            "It has circles indicating the correct differences. "
            "Please list all the differences shown by describing what visual element is circled in the drawing.",
            types.Part.from_bytes(data=data, mime_type="image/jpeg")
        ]
    )
    print("Gemini response for answer_01.jpg:")
    print(response.text)

if __name__ == '__main__':
    asyncio.run(main())
