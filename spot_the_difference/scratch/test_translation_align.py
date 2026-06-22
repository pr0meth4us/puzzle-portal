import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

combined = std.load_bgr("/Users/nicksng/code/puzzle-portal/spot_the_difference/puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)

h, w = img_a.shape[:2]

# Let's estimate translation using ECC with MOTION_TRANSLATION
gray_a = std._gray(img_a)
gray_b = std._gray(img_b)

# Run ECC translation
warp_matrix = np.eye(2, 3, dtype=np.float32)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-5)
try:
    _, warp_matrix = cv2.findTransformECC(gray_a, gray_b, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
    print(f"Translation matrix:\n{warp_matrix}")
    img_b_aligned = cv2.warpAffine(img_b, warp_matrix, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
    s_ecc = std._ssim(img_a, img_b_aligned)
    print(f"SSIM after translation ECC: {s_ecc:.4f}")
    
    # Check left edge in img_b_aligned
    gray_b_aligned = cv2.cvtColor(img_b_aligned, cv2.COLOR_BGR2GRAY)
    print("\nColumn | Mean gray_a | Mean gray_b_aligned")
    print("-" * 45)
    for x in range(20, 50, 2):
        mean_a = np.mean(gray_a[:, x])
        mean_b = np.mean(gray_b_aligned[:, x])
        print(f"x={x:2d}  | {mean_a:10.2f} | {mean_b:10.2f}")
        
except Exception as e:
    print(f"ECC translation failed: {e}")
