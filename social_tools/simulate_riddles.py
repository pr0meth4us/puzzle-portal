#!/usr/bin/env python3
import os
import sys
import glob
import time
import asyncio
from pathlib import Path

# Add current directory to path to allow importing fb_messenger_bot
sys.path.append(str(Path(__file__).resolve().parent.parent))
from social_tools.fb_messenger_bot import get_gemini_answer_from_screenshot, chat_history, FAMILY_CONTEXT

DATASET_DIR = Path(__file__).resolve().parent.parent / 'folder'

async def simulate():
    if not DATASET_DIR.exists():
        print(f"Dataset directory not found: {DATASET_DIR}")
        return

    # Find all subdirectories
    subdirs = [d for d in DATASET_DIR.iterdir() if d.is_dir()]
    subdirs.sort()

    print(f"=== Starting Offline Simulator ===")
    print(f"Dataset path: {DATASET_DIR}")
    print(f"Found {len(subdirs)} rounds/folders.")
    print(f"Family Context: {FAMILY_CONTEXT}")
    print("=" * 40)

    for round_dir in subdirs:
        print(f"\n>>>>> Starting Round: {round_dir.name} <<<<<")
        
        # Get all images in this round folder recursively
        images = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            images.extend(round_dir.rglob(ext))
        
        # Sort images lexicographically (which should order them chronologically by ID)
        images.sort()
        
        for idx, img_path in enumerate(images):
            print(f"\n--- [Image {idx+1}/{len(images)}] {img_path.name} ---")
            
            with open(img_path, 'rb') as f:
                image_bytes = f.read()
                
            print("Sending to Gemini...")
            start_time = time.time()
            
            # Use the exact same logic the bot uses in production
            answer = await get_gemini_answer_from_screenshot(image_bytes)
            
            elapsed = time.time() - start_time
            print(f"[⏱️ {elapsed:.2f}s] [GEMINI RAW OUTPUT]: {answer}")
            
            # Simulate bot behavior
            if answer == "WON":
                print("🏆 [BOT ACTION]: Typed 'អរគុណច្រើនបង! ជយោ!' and cleared history.")
                chat_history.clear()
            elif answer == "RESET":
                print("🔄 [BOT ACTION]: Puzzle ended/Lost. Clearing history.")
                chat_history.clear()
            elif answer.startswith("NEW_RIDDLE_ANSWER:"):
                actual = answer.replace("NEW_RIDDLE_ANSWER:", "").strip()
                print("✨ [BOT ACTION]: Detected completely new puzzle. Cleared old context.")
                print(f"⌨️  [BOT ACTION]: Typed '{actual}'")
                # Note: get_gemini_answer_from_screenshot already clears history internally for this
            elif answer != "IGNORE":
                print(f"⌨️  [BOT ACTION]: Typed '{answer}'")
            else:
                print("🤫 [BOT ACTION]: Did nothing (Ignored).")
                
            # Print current memory size
            print(f"[MEMORY]: Currently holding {len(chat_history)} turns in memory.")
            
            # Artificial delay between images
            time.sleep(1)
            
        print(f"\n<<<<< Finished Round: {round_dir.name} >>>>>")
        
if __name__ == "__main__":
    asyncio.run(simulate())
