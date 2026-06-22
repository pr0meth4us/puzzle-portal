import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import spot_the_differences as std

combined = std.load_bgr("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
img_b_aligned, valid_y_range, H_align = std.align(img_a, img_b, skip_ecc=False)

# Crop region around rock: x=80..200, y=100..300
crop_a = img_a[100:300, 80:200]
crop_b = img_b_aligned[100:300, 80:200]
diff = cv2.absdiff(crop_a, crop_b)

print("Diff stats for rock region:")
print("Max diff:", np.max(diff))
print("Mean diff:", np.mean(diff))

gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Number of contours in rock region: {len(contours)}")
for idx, c in enumerate(contours):
    (cx, cy), r = cv2.minEnclosingCircle(c)
    area = cv2.contourArea(c)
    global_cx = cx + 80
    global_cy = cy + 100
    print(f"Contour {idx}: local=({cx:.1f}, {cy:.1f}), global=({global_cx:.1f}, {global_cy:.1f}), r={r:.1f}, area={area:.1f}")
