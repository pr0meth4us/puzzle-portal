import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from spot_the_differences import load_bgr, crop_text_by_gap, auto_slice

def main():
    img_path = Path("puzzles/puzzle_07.jpg")
    combined = load_bgr(str(img_path))
    h, w = combined.shape[:2]
    
    cropped_combined, crop_y_offset = crop_text_by_gap(combined)
    img_a, img_b, a_start, b_start = auto_slice(cropped_combined)
    
    ha, wa = img_a.shape[:2]
    gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    
    col_diff = np.mean(np.abs(gray[:, 1:].astype(float) - gray[:, :-1].astype(float)), axis=0)
    row_diff = np.mean(np.abs(gray[1:, :].astype(float) - gray[:-1, :].astype(float)), axis=1)
    
    print("--- Left edge col_diff (indices 0 to 50) ---")
    for i in range(50):
        print(f"col_diff[{i}] = {col_diff[i]:.2f}")
        
    print("--- Right edge col_diff (indices -50 to -1) ---")
    for i in range(wa - 50, wa - 1):
        print(f"col_diff[{i}] = {col_diff[i]:.2f}")
        
    print("--- Top edge row_diff (indices 0 to 50) ---")
    for i in range(50):
        print(f"row_diff[{i}] = {row_diff[i]:.2f}")
        
    print("--- Bottom edge row_diff (indices -50 to -1) ---")
    for i in range(ha - 50, ha - 1):
        print(f"row_diff[{i}] = {row_diff[i]:.2f}")

if __name__ == "__main__":
    main()
