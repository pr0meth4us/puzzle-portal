import sys
from pathlib import Path
import cv2
import numpy as np

SCRIPT_DIR = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference")
sys.path.append(str(SCRIPT_DIR.parent))
import spot_the_difference.spot_the_differences as std

img_path = SCRIPT_DIR / "puzzles" / "puzzle_08.jpg"
combined = std.load_bgr(str(img_path))
cropped, crop_y_offset = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped)

print("Aligning...")
img_b_aligned, valid_y_range, H_align, valid_mask = std.align(img_a, img_b, skip_ecc=False)

print("Detecting...")
circles, count = std.detect(img_a, img_b_aligned,
                            min_area=50,
                            delta_floor=12.0,
                            valid_y_range=valid_y_range,
                            split_dir="horizontal",
                            H=H_align,
                            edge_mask_ksize=5,
                            merge_radius_override=40,
                            valid_mask=valid_mask)

print(f"Found {len(circles)} circles:")
diff = cv2.absdiff(img_a, img_b_aligned)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
if valid_mask is not None:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_valid = cv2.erode(valid_mask, kernel)
    gray_diff = cv2.bitwise_and(gray_diff, eroded_valid)

h, w = img_a.shape[:2]

for idx, (cx, cy, r) in enumerate(circles):
    cx, cy = int(cx), int(cy)
    r_size = max(40, int(r * 1.5))
    y1, y2 = max(0, cy - r_size), min(h, cy + r_size)
    x1, x2 = max(0, cx - r_size), min(w, cx + r_size)
    roi = gray_diff[y1:y2, x1:x2].copy()
    
    roi[0:2, :] = 0
    roi[-2:, :] = 0
    roi[:, 0:2] = 0
    roi[:, -2:] = 0
    
    max_val = np.max(roi)
    thresh_val = max(8, int(max_val * 0.25))
    _, th = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_cnts = [c for c in cnts if cv2.contourArea(c) >= 4]
    print(f"  Circle {idx+1}: center=({cx}, {cy}), r={r}, max_val={max_val}, thresh_val={thresh_val}, cnts={len(cnts)}, valid_cnts={len(valid_cnts)}")
