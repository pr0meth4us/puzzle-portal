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

gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b_aligned, cv2.COLOR_BGR2GRAY)

print("Comparing game board starting x-coordinate (where gray value > 100) at different y:")
print("y-coord | A start x | B_aligned start x | Diff")
print("-" * 50)
for y in [100, 200, 300, 421, 500, 600, 700, 800, 900]:
    # Find first x where gray value > 100
    x_a = -1
    for x in range(15, 100):
        if gray_a[y, x] > 100:
            x_a = x
            break
            
    x_b = -1
    for x in range(15, 100):
        if gray_b[y, x] > 100:
            x_b = x
            break
            
    print(f"y={y:3d}   | {x_a:9d} | {x_b:17d} | {x_b - x_a:4d}")
