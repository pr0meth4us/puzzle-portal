import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import spot_the_difference.engine_v3 as eng

p_path1 = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference/puzzles/puzzle_extra_05.jpg")
p_path2 = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference/puzzles/puzzle_extra_06.jpg")
img_a = cv2.imread(str(p_path1))
img_b = cv2.imread(str(p_path2))

# Align
img_b_aligned, H_align, valid_mask = eng.align(img_a, img_b, skip_ecc=True)

# Save an overlay image of img_a and img_b_aligned to check alignment quality
overlay = cv2.addWeighted(img_a, 0.5, img_b_aligned, 0.5, 0)
cv2.imwrite("/Users/nicksng/code/puzzle-portal/spot_the_difference/scratch/overlay_sift.png", overlay)

# Let's also do a simple resize (no homography) and check overlay
img_b_resized = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
overlay_resized = cv2.addWeighted(img_a, 0.5, img_b_resized, 0.5, 0)
cv2.imwrite("/Users/nicksng/code/puzzle-portal/spot_the_difference/scratch/overlay_resized.png", overlay_resized)

print("SSIM SIFT:", eng._ssim_score(img_a, img_b_aligned))
print("SSIM Resized:", eng._ssim_score(img_a, img_b_resized))
