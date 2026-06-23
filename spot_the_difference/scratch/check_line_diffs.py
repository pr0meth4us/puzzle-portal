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
    res_path = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference/results/res_puzzle_extra_05_vs_puzzle_extra_06.png")
    ans_path = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference/correct_answers/answer_01.jpg")
    
    with open(res_path, 'rb') as f: res_data = f.read()
    with open(ans_path, 'rb') as f: ans_data = f.read()
    
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "Compare these two images: the first is our output result where differences are circled (or should be), "
            "and the second is the correct answer key showing the ground truth differences. "
            "Please check if the circles in our output match the circles in the correct answer key. "
            "Are they completely misaligned or placed on wrong visual elements? Please describe why the user says 'the 5 vs 6 is simply wrong'.",
            types.Part.from_bytes(data=res_data, mime_type="image/png"),
            types.Part.from_bytes(data=ans_data, mime_type="image/jpeg")
        ]
    )
    print("Gemini comparison:")
    print(response.text)

if __name__ == '__main__':
    asyncio.run(main())
