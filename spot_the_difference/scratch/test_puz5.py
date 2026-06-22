import sys
from pathlib import Path
import cv2
import numpy as np

SCRIPT_DIR = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference")
sys.path.append(str(SCRIPT_DIR.parent))
import spot_the_difference.spot_the_differences as std

img_path = SCRIPT_DIR / "validation_dataset" / "puzzle_05.jpg"
combined = std.load_bgr(str(img_path))
cropped, crop_y_offset = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped)

print("Aligning...")
img_b_aligned, valid_y_range, H_align, valid_mask = std.align(img_a, img_b, skip_ecc=False)

print("\n--- Running WITH mask-roi 500,480,565,555 ---")
circles1, count1 = std.detect(img_a, img_b_aligned,
                              min_area=30,
                              delta_floor=7.0,
                              valid_y_range=valid_y_range,
                              split_dir="horizontal",
                              H=H_align,
                              mask_rois=[(500,480,565,555)],
                              merge_radius_override=30,
                              valid_mask=valid_mask)
print(f"Count: {count1}")
for i, (cx, cy, r) in enumerate(circles1):
    print(f"  Circle {i+1}: ({cx}, {cy}) r={r}")

print("\n--- Running WITHOUT mask-roi ---")
circles2, count2 = std.detect(img_a, img_b_aligned,
                              min_area=30,
                              delta_floor=7.0,
                              valid_y_range=valid_y_range,
                              split_dir="horizontal",
                              H=H_align,
                              mask_rois=[],
                              merge_radius_override=30,
                              valid_mask=valid_mask)
print(f"Count: {count2}")
for i, (cx, cy, r) in enumerate(circles2):
    print(f"  Circle {i+1}: ({cx}, {cy}) r={r}")
