import cv2
import numpy as np

img = cv2.imread("correct_answers/answer_08.jpg")
if img is None:
    print("Could not read correct_answers/answer_08.jpg")
    exit()

print(f"Shape: {img.shape}")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
mask = mask1 | mask2

cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
circles = []
for c in cnts:
    area = cv2.contourArea(c)
    if area < 50:
        continue
    (cx, cy), r = cv2.minEnclosingCircle(c)
    circles.append((int(cx), int(cy), int(r)))

print(f"Found {len(circles)} red circles/regions in answer_08.jpg:")
for i, (cx, cy, r) in enumerate(circles):
    print(f"  Region {i+1}: Center=({cx}, {cy}), Radius={r}")
