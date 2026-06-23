#!/usr/bin/env python3
"""
ocr_multimodal.py — Multimodal transcription using Gemini.
Supports: PNG, JPG, JPEG, HEIC, HEIF, PDF (multi-page)
Output:   Transcribed text (especially optimized for handwritten Khmer) saved as JSON + plain .txt/markdown
"""

import os
import sys
import gc
import io
import json
import math
import time
import base64
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add project dir to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from utils.bifrost_config import get_config

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = SCRIPT_DIR / 'results'
OUTPUT_FOLDER.mkdir(exist_ok=True)

SUPPORTED_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.heic', '.heif'}
SUPPORTED_EXTS       = SUPPORTED_IMAGE_EXTS | {'.pdf'}

# Prompt for transcription
TRANSCRIBE_PROMPT = """You are an expert document transcriber. Your task is to transcribe all text from the provided image.
The document contains Khmer and English text, including handwritten content. 

Please follow these guidelines:
1. Transcribe all text (printed and handwritten) as accurately as possible. Pay close attention to Khmer handwriting (e.g., names, phone numbers, signatures, and notes).
2. For tables, lists, or structured sections, format them cleanly using Markdown tables.
3. Preserve the layout, headers, and structure of the document where possible.
4. For signatures or hand-drawn marks, use placeholders like [ហត្ថលេខា] (Signature) or [ស្នាមមេដៃ] (Thumbprint).
5. Output ONLY the clean transcribed text/markdown. Do not add any greeting, preamble, explanations, markdown fences (like ```markdown), or conversational filler. Start directly with the transcribed text.
"""

def setup_genai_client():
    from google import genai
    
    # Try Vertex AI configuration first
    vertex_project = os.getenv("VERTEX_AI_PROJECT", "khmer-ocr-496606")
    vertex_location = os.getenv("VERTEX_AI_LOCATION", "asia-southeast1")
    
    # If BIFROST successfully pulled credentials, they will be in os.environ
    gemini_api_key = get_config('GEMINI_API_KEY', '').strip()
    
    if gemini_api_key:
        print("  -> Using Google AI Studio (API Key)")
        return genai.Client(api_key=gemini_api_key)
    else:
        print(f"  -> Using Google Cloud Vertex AI (Project: {vertex_project})")
        # Ensure credentials file is resolved for Vertex AI SDK
        creds_raw = get_config('GOOGLE_APPLICATION_CREDENTIALS', str(PROJECT_DIR / 'credentials.json'))
        creds_path = Path(creds_raw)
        if not creds_path.is_absolute():
            creds_path = (PROJECT_DIR / creds_path).resolve()
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(creds_path)
        
        return genai.Client(
            vertexai=True,
            project=vertex_project,
            location=vertex_location
        )

# ── HEIC/HEIF conversion ──────────────────────────────────────────────────────

def convert_heic_to_jpeg(heic_bytes: bytes) -> bytes:
    try:
        import pillow_heif
        from PIL import Image
        pillow_heif.register_heif_opener()
        img = Image.open(io.BytesIO(heic_bytes))
        buf = io.BytesIO()
        img.convert('RGB').save(buf, format='JPEG', quality=95)
        return buf.getvalue()
    except ImportError:
        pass

    try:
        import pyheif
        from PIL import Image
        heif_file = pyheif.read_heif(heic_bytes)
        img = Image.frombytes(
            heif_file.mode, heif_file.size, heif_file.data,
            "raw", heif_file.mode, heif_file.stride
        )
        buf = io.BytesIO()
        img.convert('RGB').save(buf, format='JPEG', quality=95)
        return buf.getvalue()
    except ImportError:
        pass

    raise RuntimeError("HEIC/HEIF conversion requires pillow-heif or pyheif.")

# ── PDF helpers ───────────────────────────────────────────────────────────────

def _pdf_page_count(pdf_path: Path) -> int:
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        n = len(doc)
        doc.close()
        return n
    except ImportError:
        pass

    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf_path)).pages)
    except ImportError:
        pass

    return 0

def pdf_page_to_jpeg_bytes(pdf_path: Path, page_num: int, dpi: int) -> bytes:
    """Rasterize a single 1-based page of a PDF and return JPEG bytes."""
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(
            str(pdf_path), dpi=dpi,
            first_page=page_num, last_page=page_num,
        )
        if pages:
            buf = io.BytesIO()
            pages[0].save(buf, format='JPEG', quality=90)
            return buf.getvalue()
    except ImportError:
        pass

    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        page_idx = page_num - 1
        pix = doc[page_idx].get_pixmap(matrix=mat, alpha=False)
        jpeg_bytes = pix.tobytes('jpeg')
        doc.close()
        return jpeg_bytes
    except ImportError:
        raise RuntimeError("PDF support requires either pdf2image+poppler or PyMuPDF.")

# ── Transcription Core ────────────────────────────────────────────────────────

def to_b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode('utf-8')

async def transcribe_image_bytes(image_bytes: bytes, client, model: str) -> str:
    """Send image bytes directly to Gemini for transcription."""
    img_b64 = to_b64(image_bytes)
    
    prompt_parts = [
        {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
        {"text": TRANSCRIBE_PROMPT},
    ]

    response = await client.aio.models.generate_content(
        model=model,
        contents=[{"parts": prompt_parts}],
    )
    
    return response.text or ""

async def transcribe_file(
    file_path: Path,
    client,
    model: str,
    dpi: int = 150,
) -> list[dict]:
    ext = file_path.suffix.lower()
    pages_out = []

    if ext in SUPPORTED_IMAGE_EXTS:
        img_bytes = file_path.read_bytes()
        if ext in ('.heic', '.heif'):
            print(f"  -> Converting {ext.upper()} -> JPEG...")
            img_bytes = convert_heic_to_jpeg(img_bytes)
        
        t0 = time.time()
        print(f"  Transcribing page 1/1 via {model}...", end=' ', flush=True)
        text = await transcribe_image_bytes(img_bytes, client, model)
        elapsed = round(time.time() - t0, 3)
        print(f"{elapsed}s  ({len(text)} chars)")
        pages_out.append({'page': 1, 'text': text, 'ocr_seconds': elapsed})
        return pages_out

    if ext == '.pdf':
        total_pages = _pdf_page_count(file_path)
        print(f"  -> PDF has {total_pages} page(s)")

        for page_num in range(1, total_pages + 1):
            label = f"page {page_num}/{total_pages}"
            print(f"  Rasterizing {label}...", end=' ', flush=True)
            t_raster = time.time()
            img_bytes = pdf_page_to_jpeg_bytes(file_path, page_num, dpi)
            print(f"done ({round(time.time() - t_raster, 2)}s)")

            print(f"  Transcribing {label} via {model}...", end=' ', flush=True)
            t_trans = time.time()
            text = await transcribe_image_bytes(img_bytes, client, model)
            elapsed = round(time.time() - t_trans, 3)
            print(f"{elapsed}s  ({len(text)} chars)")

            pages_out.append({'page': page_num, 'text': text, 'ocr_seconds': elapsed})
            img_bytes = None
            gc.collect()

        return pages_out

    raise ValueError(f"Unsupported file type: {ext}")

# ── Output Helpers ────────────────────────────────────────────────────────────

def _base_name(file_path: Path) -> str:
    return f"{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_multimodal"

def save_results(file_path: Path, pages: list[dict]):
    base = _base_name(file_path)
    json_path = OUTPUT_FOLDER / f"{base}.json"
    txt_path  = OUTPUT_FOLDER / f"{base}.md"

    # Save detailed JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source': file_path.name,
            'pages': len(pages),
            'timestamp': datetime.now().isoformat(),
            'model': 'gemini-2.5-flash',
            'results': pages
        }, f, indent=2, ensure_ascii=False)

    # Save combined markdown/txt
    with open(txt_path, 'w', encoding='utf-8') as f:
        for p in pages:
            if len(pages) > 1:
                f.write(f"\n<!-- PAGE {p['page']} of {len(pages)} -->\n")
            f.write(p['text'].strip())
            f.write('\n\n')

    return json_path, txt_path

def resolve_inputs(args_inputs: list[str]) -> list[Path]:
    paths = []
    for raw in args_inputs:
        p = Path(raw)
        if p.is_dir():
            found = sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXTS)
            paths.extend(found)
        elif p.is_file():
            if p.suffix.lower() in SUPPORTED_EXTS:
                paths.append(p)
    return paths

# ── Entry Point ───────────────────────────────────────────────────────────────

async def main_async():
    parser = argparse.ArgumentParser(
        description="Multimodal transcription using Gemini."
    )
    parser.add_argument('inputs', nargs='+', help='File(s) or folder(s) to transcribe')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for PDF rasterization (default: 150)')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash', help='Gemini model to use (default: gemini-2.5-flash)')
    args = parser.parse_args()

    files = resolve_inputs(args.inputs)
    if not files:
        print("ERROR: No valid input files found.")
        sys.exit(1)

    print(f"\nInitializing Gemini Multimodal Client...")
    client = setup_genai_client()

    for file_path in files:
        print(f"\n{'─'*55}")
        print(f"File : {file_path.name}")
        print(f"Type : {file_path.suffix.upper().lstrip('.')}")
        print(f"{'─'*55}")

        t_start = time.time()
        try:
            pages = await transcribe_file(file_path, client, args.model, args.dpi)
            total = round(time.time() - t_start, 3)
            json_path, txt_path = save_results(file_path, pages)
            
            print(f"\n  Total time : {total}s across {len(pages)} page(s)")
            print(f"  Markdown   : {txt_path}")
            print(f"  JSON       : {json_path}")
            
            # Print a quick preview of the first page
            print(f"\n{'─'*55}")
            print("Preview of Transcribed Content:")
            print(f"{'─'*55}")
            preview_lines = pages[0]['text'].strip().splitlines()
            for line in preview_lines[:30]:
                print(f"  {line}")
            if len(preview_lines) > 30:
                print(f"  ... (+{len(preview_lines)-30} more lines)")
                
        except Exception as e:
            print(f"  ERROR processing file: {e}")

if __name__ == '__main__':
    asyncio.run(main_async())
