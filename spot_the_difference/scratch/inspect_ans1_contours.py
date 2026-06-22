import cv2
import numpy as np

img = cv2.imread("correct_answers/answer_01.jpg")
if img is None:
    print("Could not read correct_answers/answer_01.jpg")
    exit()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Find green/cyan pixels (Hue around 90-110, S > 50, V > 50)
mask = (h >= 40) & (h <= 110) & (s > 50) & (v > 50)
mask_img = np.zeros_like(h, dtype=np.uint8)
mask_img[mask] = 255

cnts, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
circles = []
for c in cnts:
    area = cv2.contourArea(c)
    if area < 30:
        continue
    (cx, cy), r = cv2.minEnclosingCircle(c)
    circles.append((int(cx), int(cy), int(r), area))

print(f"Found {len(circles)} green/cyan circles in answer_01.jpg:")
for i, (cx, cy, r, area) in enumerate(circles):
    print(f"  Circle {i+1}: center=({cx}, {cy}), r={r}, area={area}")
