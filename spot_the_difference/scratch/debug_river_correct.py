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

# Correct y-coordinate in img_a is 457.4 - 36 = 421.4
cx, cy = 45, 421
x1, x2 = max(0, cx - 50), min(img_a.shape[1], cx + 50)
y1, y2 = max(0, cy - 50), min(img_a.shape[0], cy + 50)

crop_a = img_a[y1:y2, x1:x2]
crop_b = img_b_aligned[y1:y2, x1:x2]

# Save these crops
cv2.imwrite("/Users/nicksng/code/puzzle-portal/spot_the_difference/scratch/crop_river_a_correct.png", crop_a)
cv2.imwrite("/Users/nicksng/code/puzzle-portal/spot_the_difference/scratch/crop_river_b_aligned_correct.png", crop_b)

diff = cv2.absdiff(crop_a, crop_b)
cv2.imwrite("/Users/nicksng/code/puzzle-portal/spot_the_difference/scratch/crop_river_diff_correct.png", diff)

print("Crops saved successfully!")
print(f"Crop shape: {crop_a.shape}")
print(f"Max BGR diff in crop: {np.max(diff)}")
print(f"Mean BGR diff in crop: {np.mean(diff):.2f}")

# Let's print out the BGR diff at the center of the crop
print("\ny-coord (relative to img_a) | img_a BGR | img_b_aligned BGR | Abs Diff")
print("-" * 65)
for y in range(cy - 10, cy + 10):
    val_a = img_a[y, cx].tolist()
    val_b = img_b_aligned[y, cx].tolist()
    d = [abs(val_a[i] - val_b[i]) for i in range(3)]
    print(f"y={y:3d}                       | {val_a} | {val_b} | {d} (mean: {np.mean(d):.1f})")
