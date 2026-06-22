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

# Let's save a visualization of the left part of the panel
# We will save the first 200 columns for y from 0 to 1006.
cv2.imwrite("/Users/nicksng/code/puzzle-portal/spot_the_difference/scratch/left_edge_a.png", img_a[:, :200])
cv2.imwrite("/Users/nicksng/code/puzzle-portal/spot_the_difference/scratch/left_edge_b_aligned.png", img_b_aligned[:, :200])

# Let's also print the standard deviation or mean of pixel values along columns to see if there's a clear border boundary
gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b_aligned, cv2.COLOR_BGR2GRAY)

print("Column | Mean gray_a | Mean gray_b_aligned | Column Var gray_a | Column Var gray_b")
print("-" * 75)
for x in range(0, 150, 10):
    mean_a = np.mean(gray_a[:, x])
    mean_b = np.mean(gray_b[:, x])
    var_a = np.var(gray_a[:, x])
    var_b = np.var(gray_b[:, x])
    print(f"x={x:3d}  | {mean_a:10.2f} | {mean_b:17.2f} | {var_a:17.2f} | {var_b:17.2f}")
