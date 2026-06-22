import sys
from pathlib import Path
sys.path.append("/Users/nicksng/code/puzzle-portal")
import spot_the_difference.spot_the_differences as std
import cv2
import numpy as np

# Load extra 5 and 6
img_a = std.load_bgr("spot_the_difference/puzzles/puzzle_extra_05.jpg")
img_b = std.load_bgr("spot_the_difference/puzzles/puzzle_extra_06.jpg")

img_b_aligned, valid_y_range, H_align, valid_mask = std.align(img_a, img_b, skip_ecc=False)

circles, count = std.detect(img_a, img_b_aligned,
                            min_area=50,
                            delta_floor=7.0,
                            valid_y_range=valid_y_range,
                            split_dir=None,
                            H=H_align,
                            edge_mask_ksize=-1,
                            valid_mask=valid_mask)

print("Our Detections:")
for i, (cx, cy, r) in enumerate(circles):
    print(f"  Circle {i+1}: center=({cx}, {cy}), r={r}")

# Load answer_01.jpg
gt_img = cv2.imread("spot_the_difference/correct_answers/answer_01.jpg")
hsv = cv2.cvtColor(gt_img, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)
cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

gt_circles = []
for c in cnts:
    area = cv2.contourArea(c)
    if area < 10: continue
    (cx, cy), r = cv2.minEnclosingCircle(c)
    gt_circles.append((int(cx), int(cy), int(r)))

gt_circles.sort(key=lambda x: (x[1], x[0]))
print("\nGround Truth Circles in answer_01.jpg:")
for i, (cx, cy, r) in enumerate(gt_circles):
    print(f"  GT {i+1}: center=({cx}, {cy}), r={r}")
