#!/usr/bin/env python3
import os
import sys
import time
import asyncio
import random
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

# Import logic from the different architectures
from social_tools.fb_messenger_bot import get_gemini_answer_from_screenshot
from ocr_tools.main import setup as ocr_setup, process as ocr_process

DATASET_DIR = PROJECT_DIR / 'folder'
ROUTER_SCRIPT = PROJECT_DIR / 'puzzle_portal' / 'router.py'

async def run_benchmark():
    if not DATASET_DIR.exists():
        print(f"Dataset not found at {DATASET_DIR}")
        return

    images = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPEG'):
        images.extend(DATASET_DIR.rglob(ext))
        
    if not images:
        print("No images found in dataset.")
        return
        
    # Use first 10 images for faster testing
    test_images = images[:10]
    sample_size = len(test_images)
    
    print(f"=== Starting Benchmark on First {sample_size} Images ===\n")
    
    # Initialize the heavy clients (Cloud Vision & GenAI)
    vision_client, genai_client = ocr_setup()
    
    results = []
    
    for idx, img_path in enumerate(test_images):
        print(f"[{idx+1}/{sample_size}] Evaluating: {img_path.name}")
        with open(img_path, 'rb') as f:
            image_bytes = f.read()
            
        # ---------------------------------------------------------
        # Approach 1: Direct Vision (Messenger Bot)
        # ---------------------------------------------------------
        start_t = time.time()
        try:
            # We clear chat_history manually to keep it fair (no memory bleed)
            from social_tools.fb_messenger_bot import chat_history
            chat_history.clear()
            
            ans1 = await get_gemini_answer_from_screenshot(image_bytes)
        except Exception as e:
            ans1 = f"ERROR: {e}"
        time1 = time.time() - start_t
        
        # ---------------------------------------------------------
        # Approach 2: Cloud OCR + Text LLM (ocr_tools)
        # ---------------------------------------------------------
        start_t = time.time()
        try:
            res2 = await ocr_process(img_path, vision_client, genai_client)
            ans2 = res2['parsed_answer'].get('final_answer', 'NO_ANSWER')
        except Exception as e:
            ans2 = f"ERROR: {e}"
        time2 = time.time() - start_t
        
        # ---------------------------------------------------------
        # Approach 3: Puzzle Portal Router (Subprocess Workflow)
        # ---------------------------------------------------------
        start_t = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(ROUTER_SCRIPT), str(img_path)],
                capture_output=True, text=True, check=False
            )
            out = proc.stdout
            if "Routing to" in out:
                lines = [l for l in out.split('\n') if "Routing to" in l]
                ans3 = lines[-1].split("Routing to ")[-1].strip()
            else:
                ans3 = "ROUTER_ERROR"
        except Exception as e:
            ans3 = f"ERROR: {e}"
        time3 = time.time() - start_t
        
        results.append({
            'name': img_path.name,
            'path': img_path,
            'a1_time': time1, 'a1_ans': ans1,
            'a2_time': time2, 'a2_ans': ans2,
            'a3_time': time3, 'a3_ans': ans3
        })
        
    # Print the beautiful comparison table
    print("\n\n" + "=" * 125)
    print(f"{'Image ID':<25} | {'A1: Direct Vision (Bot)':<30} | {'A2: Cloud OCR + LLM':<30} | {'A3: Portal Router':<30}")
    print("-" * 125)
    
    avg_t1 = sum(r['a1_time'] for r in results) / sample_size if sample_size else 0
    avg_t2 = sum(r['a2_time'] for r in results) / sample_size if sample_size else 0
    avg_t3 = sum(r['a3_time'] for r in results) / sample_size if sample_size else 0
    
    for r in results:
        # truncate long answers for table formatting
        a1_trunc = (r['a1_ans'][:20] + '..').replace('\n', ' ') if len(r['a1_ans']) > 20 else r['a1_ans'].replace('\n', ' ')
        a2_trunc = (r['a2_ans'][:20] + '..').replace('\n', ' ') if len(r['a2_ans']) > 20 else r['a2_ans'].replace('\n', ' ')
        a3_trunc = (r['a3_ans'][:20] + '..').replace('\n', ' ') if len(r['a3_ans']) > 20 else r['a3_ans'].replace('\n', ' ')
        
        # Combine latency and answer
        s1 = f"[{r['a1_time']:.2f}s] {a1_trunc}"
        s2 = f"[{r['a2_time']:.2f}s] {a2_trunc}"
        s3 = f"[{r['a3_time']:.2f}s] {a3_trunc}"
        
        print(f"{r['name']:<25} | {s1:<30} | {s2:<30} | {s3:<30}")
        
    print("-" * 125)
    print(f"{'AVERAGE LATENCY':<25} | [{avg_t1:.2f}s]{' ':<23} | [{avg_t2:.2f}s]{' ':<23} | [{avg_t3:.2f}s]")
    print("=" * 125)
    
    # Write full markdown report
    artifact_dir = Path('/Users/nicksng/.gemini/antigravity-ide/brain/9f155d11-fb38-46f5-b63a-2a610a135339')
    report_path = artifact_dir / 'full_benchmark_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Full Benchmark Report (All Datasets)\n\n")
        f.write("This report contains the **full, un-truncated answers** for every image tested across all 3 approaches.\n\n")
        
        for i, r in enumerate(results):
            # Rewrite path to artifact directory so UI renders it correctly
            rel_path = r['path'].relative_to(DATASET_DIR)
            art_img_path = artifact_dir / rel_path
            
            f.write(f"## {i+1}. {r['name']}\n")
            f.write(f"![{r['name']}](file://{art_img_path.absolute()})\n\n")
            
            a1_text = r['a1_ans']
            # If A1 returned an image response, copy it to artifact dir and render it
            if "[IMAGE_RESPONSE:" in a1_text:
                import shutil
                start_idx = a1_text.find("[IMAGE_RESPONSE:") + len("[IMAGE_RESPONSE:")
                end_idx = a1_text.find("]", start_idx)
                if end_idx != -1:
                    img_path = a1_text[start_idx:end_idx].strip()
                    try:
                        # Copy to artifact directory to make it renderable
                        out_name = f"a1_res_{r['name']}"
                        shutil.copy(img_path, artifact_dir / out_name)
                        renderable_path = artifact_dir / out_name
                        # Replace the tag with a markdown image tag
                        a1_text = a1_text[:start_idx-len("[IMAGE_RESPONSE:")] + f"\n\n![A1 Image](file://{renderable_path.absolute()})\n\n" + a1_text[end_idx+1:]
                    except Exception as e:
                        print(f"Error copying image: {e}")

            f.write(f"**A1: Hybrid Direct Vision (Bot) [{r['a1_time']:.2f}s]:**\n")
            f.write(f"{a1_text}\n\n")
            f.write(f"**A2: Cloud OCR + LLM [{r['a2_time']:.2f}s]:**\n")
            f.write(f"```\n{r['a2_ans']}\n```\n\n")
            f.write(f"**A3: Portal Router [{r['a3_time']:.2f}s]:**\n")
            f.write(f"```\n{r['a3_ans']}\n```\n\n")
            f.write("---\n\n")
            
    print(f"Full markdown report saved to {report_path}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
