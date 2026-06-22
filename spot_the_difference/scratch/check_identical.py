import cv2
import numpy as np
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

puz = cv2.imread("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(puz)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
img_b_aligned, _, _ = std.align(img_a, img_b, skip_ecc=False)

left_a = img_a[400:600, 0:180]
left_b = img_b_aligned[400:600, 0:180]

diff = cv2.absdiff(left_a, left_b)
mean_diff = np.mean(diff)
max_diff = np.max(diff)

print(f"Left-edge crop absolute difference: mean={mean_diff:.4f}, max={max_diff}")
