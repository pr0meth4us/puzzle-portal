#!/usr/bin/env python3
import os
import sys
import base64
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add project dir to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from utils.bifrost_config import get_config

load_dotenv()

PROMPT = """You are an expert document transcriber. Your task is to transcribe all text from the provided image.
The document contains Khmer and English text, including handwritten content.

Please transcribe the table on page 1 of the attendance sheet.
The table has columns:
1. ល.រ (No.)
2. នាមត្រកូល - នាមខ្លួន (Last Name - First Name)
3. ភេទ (Gender)
4. តួនាទី / មុខងារ (Role / Position)
5. ឈ្មោះអាជីវកម្ម / អង្គភាព / ស្ថាប័ន (Company / Organization / Institution Name)
6. លេខទូរសព្ទ (Phone Number)
7. ហត្ថលេខា (Signature)

Format the output strictly as a Markdown table. Do not include signature images, just write [ហត្ថលេខា] or the signature text.
Do not write any introductory or concluding text, only the markdown table.
"""

def setup_client():
    from google import genai
    gemini_api_key = get_config('GEMINI_API_KEY', '').strip()
    return genai.Client(api_key=gemini_api_key)

async def test_model(client, model_name, img_b64):
    print(f"\n--- Testing model: {model_name} ---")
    try:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=[
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                {"text": PROMPT}
            ]
        )
        print(response.text)
    except Exception as e:
        print(f"Error: {e}")

async def main():
    client = setup_client()
    img_path = SCRIPT_DIR / "results" / "page1.jpg"
    img_bytes = img_path.read_bytes()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    await test_model(client, "gemini-2.5-flash", img_b64)
    await test_model(client, "gemini-2.5-pro", img_b64)

if __name__ == '__main__':
    asyncio.run(main())
