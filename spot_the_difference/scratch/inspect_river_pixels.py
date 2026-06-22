import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import spot_the_differences as std

combined = std.load_bgr("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
img_b_aligned, valid_y_range, H_align = std.align(img_a, img_b, skip_ecc=False)

# Let's print out the pixel values along y=450 (which is in the river area) for x=0..100
y = 450
print("x-coord | img_a (BGR) | img_b_aligned (BGR) | Abs Diff")
print("-" * 55)
for x in range(0, 120, 10):
    val_a = img_a[y, x].tolist()
    val_b = img_b_aligned[y, x].tolist()
    diff = [abs(val_a[i] - val_b[i]) for i in range(3)]
    print(f"x={x:3d}   | {val_a} | {val_b} | {diff} (mean: {np.mean(diff):.1f})")
