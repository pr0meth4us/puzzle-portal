import cv2
import numpy as np
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SCRIPT_DIR))
import spot_the_differences as std

for puz_name in ["puzzle_07.jpg", "puzzle_08.jpg"]:
    p_path = SCRIPT_DIR / "puzzles" / puz_name
    combined = cv2.imread(str(p_path))
    cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
    img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
    
    gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Calculate row differences
    row_diff = np.mean(np.abs(gray[1:, :].astype(float) - gray[:-1, :].astype(float)), axis=1)
    
    print(f"\n=== Adjacent Row Differences for {puz_name} ===")
    print("Top 100 rows:")
    print(row_diff[:100].round(1))
    print("Bottom 100 rows:")
    print(row_diff[-100:].round(1))
