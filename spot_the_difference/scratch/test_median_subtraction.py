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

# Standard difference maps
gray_a = std._gray(img_a)
gray_b = std._gray(img_b_aligned)

diff_gray = cv2.absdiff(gray_a, gray_b)

# Compute column-wise median along the y-axis
# We only use rows within valid_y_range to avoid text/watermarks at the top/bottom
vy0, vy1 = valid_y_range if valid_y_range else (0, h)
col_medians = np.median(diff_gray[vy0:vy1, :], axis=0)

# Subtract col_medians from diff_gray
diff_subtracted = np.clip(diff_gray.astype(float) - col_medians[None, :], 0, 255).astype(np.uint8)

# Let's check the difference values at y=421 (river patch) in both original and subtracted diff maps
print("x-coord | Original Diff | Column Median | Subtracted Diff")
print("-" * 55)
for x in range(25, 55):
    val_orig = diff_gray[421, x]
    val_med = col_medians[x]
    val_sub = diff_subtracted[421, x]
    print(f"x={x:2d}     | {val_orig:13d} | {val_med:13.1f} | {val_sub:15d}")
