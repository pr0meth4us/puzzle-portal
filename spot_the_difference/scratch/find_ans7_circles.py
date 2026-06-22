import cv2
import numpy as np

img = cv2.imread("correct_answers/answer_07.jpg")
if img is None:
    print("Could not read image.")
    exit()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
mask = mask1 | mask2
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(cnts)} red circles/regions in answer_07.jpg:")
count = 0
for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    if area < 50:
        continue
    count += 1
    (cx, cy), r = cv2.minEnclosingCircle(c)
    print(f"  GT {count}: Center=({cx:.1f}, {cy:.1f}), r={r:.1f}")
