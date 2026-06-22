import cv2
import numpy as np

puz = cv2.imread("puzzles/puzzle_07.jpg")
ans = cv2.imread("correct_answers/answer_07.jpg")

print(f"Puzzle size: {puz.shape if puz is not None else 'None'}")
print(f"Answer size: {ans.shape if ans is not None else 'None'}")

# Detect red circles in Answer 7
hsv = cv2.cvtColor(ans, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
red_mask = mask1 | mask2

cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Raw red contours count: {len(cnts)}")

dots = []
for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * area / (perimeter ** 2)
    if area > 20:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        dots.append((cx, cy, r, area, circularity))

dots.sort(key=lambda x: x[1])
print(f"Filtered red circles (sorted by Y):")
for i, (cx, cy, r, area, circ) in enumerate(dots):
    print(f"  Red Circle {i+1}: Center=({cx:.1f}, {cy:.1f}), r={r:.1f}, Area={area:.1f}, Circ={circ:.3f}")
