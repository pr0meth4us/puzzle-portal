import cv2
import numpy as np

img = cv2.imread("correct_answers/answer_05.jpg")
if img is None:
    print("Could not read correct_answers/answer_05.jpg")
    exit()

# The answer image has red circles drawn on it.
# Let's find them by thresholding the red channel/HSV space.
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
mask = mask1 | mask2

# Apply closing to merge the red strokes
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
circles = []
for c in cnts:
    area = cv2.contourArea(c)
    if area < 50:
        continue
    (cx, cy), r = cv2.minEnclosingCircle(c)
    circles.append((int(cx), int(cy), int(r)))

print(f"Found {len(circles)} red circles in answer_05.jpg:")
for i, (cx, cy, r) in enumerate(circles):
    print(f"  Circle {i+1}: center=({cx}, {cy}), r={r}")
