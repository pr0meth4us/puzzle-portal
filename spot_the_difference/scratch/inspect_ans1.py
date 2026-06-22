import cv2
import numpy as np

img = cv2.imread("correct_answers/answer_01.jpg")
if img is None:
    print("Could not read correct_answers/answer_01.jpg")
    exit()

print(f"Shape: {img.shape}")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
mask = mask1 | mask2

cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
red_count = 0
for c in cnts:
    if cv2.contourArea(c) > 20:
        red_count += 1

print(f"Number of red contours: {red_count}")
