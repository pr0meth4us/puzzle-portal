import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

combined = std.load_bgr("/Users/nicksng/code/puzzle-portal/spot_the_difference/puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
img_b_aligned, valid_y_range, H_align = std.align(img_a, img_b, skip_ecc=False)

h, w = img_a.shape[:2]

gray_a = std._gray(img_a)
gray_b = std._gray(img_b_aligned)

diff = cv2.absdiff(gray_a, gray_b)

# We will downsample vertically by 20 and horizontally by 5
sh = h // 20
sw = 60 // 5
downsampled = np.zeros((sh, sw), dtype=np.uint8)

for r in range(sh):
    for c in range(sw):
        downsampled[r, c] = np.mean(diff[r*20:(r+1)*20, c*5:(c+1)*5])

print("Full Left Edge Difference Map (y downsampled 20x, x downsampled 5x):")
print("    " + "".join(f"{c*5:4d}" for c in range(sw)))
print("-" * 60)
for r in range(sh):
    row_str = f"y={r*20:3d} |"
    for c in range(sw):
        val = downsampled[r, c]
        if val < 5:
            row_str += "    "
        elif val < 15:
            row_str += " .  "
        elif val < 30:
            row_str += " *  "
        elif val < 60:
            row_str += " #  "
        else:
            row_str += " @  "
    print(row_str)
