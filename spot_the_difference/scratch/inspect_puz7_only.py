import cv2
import numpy as np

ans = cv2.imread("correct_answers/answer_07.jpg")
print(f"Answer shape: {ans.shape}")

hsv = cv2.cvtColor(ans, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = mask1 | mask2

cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
dots = []
for c in cnts:
    area = cv2.contourArea(c)
    if area > 5:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if 4 < r < 30:
            dots.append((cx, cy, r, area))

filtered_dots = []
for cx, cy, r, area in dots:
    too_close = False
    for kx, ky, _, _ in filtered_dots:
        if np.hypot(cx - kx, cy - ky) < 10:
            too_close = True
            break
    if not too_close:
        filtered_dots.append((cx, cy, r, area))

filtered_dots.sort(key=lambda x: x[1])
print(f"Unique red dots: {len(filtered_dots)}")
for i, (cx, cy, r, area) in enumerate(filtered_dots):
    print(f"  Dot {i+1}: Center=({cx:.2f}, {cy:.2f}), r={r:.2f}, Area={area:.2f}")
