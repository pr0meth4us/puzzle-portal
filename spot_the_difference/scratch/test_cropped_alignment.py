import cv2
import numpy as np
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

puz = cv2.imread("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(puz)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)

# Crop left 36 pixels
img_a_cropped = img_a[:, 36:]
img_b_cropped = img_b[:, 36:]

# Align cropped panels
img_b_aligned, valid_y_range, H_align = std.align(img_a_cropped, img_b_cropped, skip_ecc=False)

ssim_val = std._ssim(img_a_cropped, img_b_aligned)
print(f"SSIM after aligned cropping: {ssim_val:.4f}")
print(f"H_align:\n{H_align}")
