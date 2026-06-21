import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
img = cv2.imread(str(SCRIPT_DIR / "correct_answers/answer_06.jpg"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)
cnts, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

gx, gy, gw, gh = 75, 28, 641, 1397
panel_h = gh / 2  # 698.5

rows, cols = 10, 9
cell_w = gw / cols
cell_h = panel_h / rows

top_circles = []
bottom_circles = []

for c in cnts:
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    if area < 50 or perimeter == 0:
        continue
    circularity = 4 * np.pi * area / (perimeter ** 2)
    # Check if circular and of the right size for a marking circle
    if circularity > 0.65:
        x, y, w, h = cv2.boundingRect(c)
        if 50 < w < 85 and 50 < h < 85:
            cx, cy = x + w/2, y + h/2
            # Check if in grid A (top)
            if gx <= cx <= gx+gw:
                if gy <= cy <= gy+panel_h:
                    col = int((cx - gx) / cell_w) + 1
                    row = int((cy - gy) / cell_h) + 1
                    top_circles.append((row, col))
                elif gy+panel_h <= cy <= gy+gh:
                    col = int((cx - gx) / cell_w) + 1
                    row = int((cy - (gy + panel_h)) / cell_h) + 1
                    bottom_circles.append((row, col))

top_circles = sorted(list(set(top_circles)))
bottom_circles = sorted(list(set(bottom_circles)))

print(f"Top panel circles (count: {len(top_circles)}):")
for r, c in top_circles:
    print(f"  Row {r}, Col {c}")

print(f"\nBottom panel circles (count: {len(bottom_circles)}):")
for r, c in bottom_circles:
    print(f"  Row {r}, Col {c}")
