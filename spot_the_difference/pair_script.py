import os
import asyncio
from pathlib import Path
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv('/Users/nicksng/code/random/.env')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)

async def describe_dir(dir_path):
    results = {}
    for img_path in Path(dir_path).glob('*.*'):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']: continue
        with open(img_path, 'rb') as f:
            data = f.read()
        mime_type = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Describe this image briefly. If it's a puzzle, describe the subject. If it's an answer key showing differences, mention the subject and exactly how many differences are circled or indicated.",
                types.Part.from_bytes(data=data, mime_type=mime_type)
            ]
        )
        results[img_path.name] = response.text.strip()
        print(f"[{Path(dir_path).name}] {img_path.name}: {response.text.strip()[:100]}...")
    return results

async def main():
    print("Describing Correct Answers...")
    ans = await describe_dir('/Users/nicksng/code/random/spot_the_difference/correct answers')
    print("\nDescribing Puzzles...")
    puz = await describe_dir('/Users/nicksng/code/random/spot_the_difference/puzzles')

if __name__ == '__main__':
    asyncio.run(main())
