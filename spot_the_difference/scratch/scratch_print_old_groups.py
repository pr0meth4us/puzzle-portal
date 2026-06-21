import cv2
import numpy as np
import spot_the_differences

combined = spot_the_differences.load_bgr("validation_dataset/puzzle_03.jpg")
cropped, crop_y = spot_the_differences.crop_text_by_gap(combined)
img_a, img_b, _, _ = spot_the_differences.auto_slice(cropped)
img_b_aligned, valid_y_range, H_align = spot_the_differences.align(img_a, img_b)

# Run detection with bx=16, by=8
split_dir = "horizontal"
circles, count = spot_the_differences.detect(
    img_a, img_b_aligned,
    min_area=50,
    delta_floor=7.0,
    valid_y_range=valid_y_range,
    split_dir=split_dir,
    H=H_align
)

print(f"Groups found: {len(circles)}")
for idx, (cx, cy, r) in enumerate(circles):
    print(f"Group {idx}: cx={cx}, cy={cy}, r={r}")
