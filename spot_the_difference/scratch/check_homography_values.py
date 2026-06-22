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

print(f"Homography matrix H:\n{H_align}")

# Let's map a few points from B to A
# Panel B points: (x, y)
points = np.float32([
    [0, 0],
    [100, 100],
    [500, 500],
    [1152, 1006]
]).reshape(-1, 1, 2)

if H_align is not None:
    mapped = cv2.perspectiveTransform(points, H_align).reshape(-1, 2)
    print("\nPoint Mapping (B -> A):")
    for pt, m_pt in zip(points.reshape(-1, 2), mapped):
        diff = m_pt - pt
        print(f"  B: {pt} -> A: {m_pt} | Shift: dx={diff[0]:.2f}, dy={diff[1]:.2f}")
else:
    print("H_align is None!")
