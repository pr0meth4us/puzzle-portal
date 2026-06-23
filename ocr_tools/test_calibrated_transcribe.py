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

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

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

--- CRITICAL CALIBRATION ---
To help you calibrate to the handwriting style, here are the correct transcriptions for the first two rows:
Row 1:
- ល.រ: 1
- នាមត្រកូល - នាមខ្លួន: នៅ ណាត
- ភេទ: ស្រី
- តួនាទី / មុខងារ: កាន់ស្តុក
- ឈ្មោះអាជីវកម្ម / អង្គភាព / ស្ថាប័ន: ក្រុមហ៊ុន ហេងលី
- លេខទូរសព្ទ: 0964355861
- ហត្ថលេខា: [ហត្ថលេខា]

Row 2:
- ល.រ: 2
- នាមត្រកូល - នាមខ្លួន: ហេង ដានីត
- ភេទ: ស្រី
- តួនាទី / មុខងារ: ជំនួយការគណនេយ្យ
- ឈ្មោះអាជីវកម្ម / អង្គភាព / ស្ថាប័ន: ក្រុមហ៊ុន ហេងលី
- លេខទូរសព្ទ: 0964007679
- ហត្ថលេខា: [ហត្ថលេខា]

Observe the handwriting style of these correct entries and use them to transcribe Rows 3, 4, 5, and 6. 
Do NOT guess names or companies based on the printed header logos (like Ministry of Economy and Finance, SDF, Techo Startup Center). Look only at the actual handwritten text in the cells.

Format the output strictly as a Markdown table. Do not write any introductory or concluding text, only the markdown table.
"""

def setup_gemini():
    from google import genai
    gemini_api_key = get_config('GEMINI_API_KEY', '').replace('\ufeff', '').strip()
    return genai.Client(api_key=gemini_api_key)

def setup_openai():
    from openai import AsyncOpenAI
    openai_api_key = get_config('OPENAI_API_KEY', '').replace('\ufeff', '').strip()
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY", "").replace('\ufeff', '').strip()
    return AsyncOpenAI(api_key=openai_api_key)

async def test_gemini(client, model_name, img_b64):
    print(f"\n--- Testing Gemini model: {model_name} ---")
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

async def test_openai(client, model_name, img_b64):
    print(f"\n--- Testing OpenAI model: {model_name} ---")
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096,
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

async def main():
    # Rasterize page 1 at 300 DPI
    from ocr_multimodal import pdf_page_to_jpeg_bytes
    pdf_path = Path('/Users/nicksng/code/puzzle-portal/ocr_tools/reports/13-June-2026.pdf')
    jpeg_bytes = pdf_page_to_jpeg_bytes(pdf_path, 1, 300)
    img_b64 = base64.b64encode(jpeg_bytes).decode('utf-8')
    
    gemini_client = setup_gemini()
    openai_client = setup_openai()
    
    await test_gemini(gemini_client, "gemini-2.5-pro", img_b64)
    await test_openai(openai_client, "gpt-4o", img_b64)

if __name__ == '__main__':
    asyncio.run(main())
