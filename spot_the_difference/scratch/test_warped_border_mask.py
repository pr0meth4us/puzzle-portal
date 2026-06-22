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

# Initialize mask for A
bmask = np.ones((h, w), dtype=np.uint8) * 255
std._mask_margins(img_a, bmask)

# Initialize mask for B and detect margins before warping
bmask_b = np.ones(img_b.shape[:2], dtype=np.uint8) * 255
std._mask_margins(img_b, bmask_b)

if H_align is not None:
    warped_bmask_b = cv2.warpPerspective(bmask_b, H_align, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    warped_bmask_b = cv2.erode(warped_bmask_b, kernel)
    bmask = cv2.bitwise_and(bmask, warped_bmask_b)

# Print comparison of variance or mask value along the left side
print("Mask values along y=450:")
print("x-coord | bmask_a | warped_bmask_b | final bmask")
print("-" * 50)
for x in range(0, 80, 5):
    val_a = bmask[450, x]
    print(f"x={x:3d}   | {val_a}")
