import sys
import cv2
from spot_the_differences import auto_slice, align, detect
img = cv2.imread('puzzles/puzzle6.jpg')
img_a, img_b, _, _ = auto_slice(img)
img_b_aligned, valid_y_range, _ = align(img_a, img_b, skip_ecc=True)
circles, count = detect(img_a, img_b_aligned, min_area=50, delta_floor=7.0, valid_y_range=valid_y_range)
print(f"Count: {count}")
