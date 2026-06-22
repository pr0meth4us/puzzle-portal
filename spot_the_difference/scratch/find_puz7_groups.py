import sys
from pathlib import Path
import cv2
import numpy as np

SCRIPT_DIR = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference")
sys.path.append(str(SCRIPT_DIR.parent))
import spot_the_difference.spot_the_differences as std

img_path = SCRIPT_DIR / "puzzles" / "puzzle_07.jpg"
combined = std.load_bgr(str(img_path))
cropped, crop_y_offset = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped)

print("Aligning...")
img_b_aligned, valid_y_range, H_align, valid_mask = std.align(img_a, img_b, skip_ecc=False)

print("Detecting...")
circles, count = std.detect(img_a, img_b_aligned,
                            min_area=30,
                            delta_floor=8.0,
                            valid_y_range=valid_y_range,
                            split_dir="horizontal",
                            H=H_align,
                            edge_mask_ksize=5,
                            custom_left_margin=48,
                            mask_rois=[(810,630,860,680), (880,860,930,910)],
                            merge_radius_override=55,
                            valid_mask=valid_mask)

print(f"Found {len(circles)} circles:")
for i, (cx, cy, r) in enumerate(circles):
    print(f"  Circle {i+1}: center=({cx}, {cy}), r={r}")
