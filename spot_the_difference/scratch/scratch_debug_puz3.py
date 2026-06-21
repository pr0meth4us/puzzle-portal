import cv2
import numpy as np
import spot_the_differences

combined = spot_the_differences.load_bgr("validation_dataset/puzzle_03.jpg")
img_a, img_b, _, _ = spot_the_differences.auto_slice(combined)
img_b_aligned, _, H = spot_the_differences.align(img_a, img_b)

diff = cv2.absdiff(img_a, img_b_aligned)
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Raw contours: {len(contours)}")
for i, c in enumerate(contours):
    if cv2.contourArea(c) > 20:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        print(f"Contour {i}: cx={int(cx)}, cy={int(cy)}, r={int(r)}, area={cv2.contourArea(c)}")
