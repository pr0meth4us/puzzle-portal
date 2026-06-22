import cv2
import numpy as np

img = cv2.imread("puzzles/puzzle_08.jpg")
if img is None:
    print("Could not read puzzle_08.jpg")
    exit()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
mask = mask1 | mask2

cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
red_circles = []
for c in cnts:
    area = cv2.contourArea(c)
    if area < 100:
        continue
    peri = cv2.arcLength(c, True)
    circ = 4 * np.pi * area / (peri**2) if peri > 0 else 0
    if circ > 0.6:
        red_circles.append((c, area, circ))

print(f"Total red pixels: {np.sum(mask == 255)}")
print(f"Found {len(red_circles)} red circular shapes in puzzle_08.jpg")
