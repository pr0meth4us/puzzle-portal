import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent

for puz_name in ["puzzle_07.jpg", "puzzle_08.jpg"]:
    p_path = SCRIPT_DIR / "puzzles" / puz_name
    img = cv2.imread(str(p_path))
    if img is None: continue
    h, w = img.shape[:2]
    half = h // 2
    img_a = img[:half]
    gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    
    # Calculate difference between adjacent columns
    col_diff = np.mean(np.abs(gray[:, 1:].astype(float) - gray[:, :-1].astype(float)), axis=0)
    
    print(f"\n=== Adjacent Column Differences for {puz_name} ===")
    print("First 100 columns:")
    print(col_diff[:100].round(3))
    print("Last 100 columns:")
    print(col_diff[-100:].round(3))
