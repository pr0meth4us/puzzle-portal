import time
import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER = SCRIPT_DIR / 'khmer_test_images'
MAIN_SCRIPT = SCRIPT_DIR / 'main.py'

def get_files_state():
    state = {}
    if IMAGE_FOLDER.exists():
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            for f in IMAGE_FOLDER.glob(ext):
                try:
                    state[f] = f.stat().st_mtime
                except FileNotFoundError:
                    # File might have been deleted right after glob found it
                    pass
    return state

def main():
    print(f"Watching {IMAGE_FOLDER} for new images...")
    current_state = get_files_state()
    
    try:
        while True:
            time.sleep(1)  # Poll every 1 second
            new_state = get_files_state()
            
            # Check if there are new files or modified files
            added_or_modified = []
            for f, mtime in new_state.items():
                if f not in current_state or current_state[f] < mtime:
                    added_or_modified.append(f)
                    
            if added_or_modified:
                print(f"\n[{time.strftime('%X')}] Detected new/modified image(s): {[f.name for f in added_or_modified]}")
                print(f"Triggering {MAIN_SCRIPT.name}...")
                try:
                    subprocess.run([sys.executable, str(MAIN_SCRIPT)])
                except Exception as e:
                    print(f"Error running main.py: {e}")
                
            current_state = new_state
            
    except KeyboardInterrupt:
        print("\nStopping watcher.")

if __name__ == '__main__':
    main()
