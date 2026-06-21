import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
img = cv2.imread(str(SCRIPT_DIR / "correct_answers/answer_06.jpg"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to find lines/circles
# Grid lines and text are dark, background is light.
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)

cnts, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours: {len(cnts)}")

circular_contours = []
for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    if area < 50 or perimeter == 0:
        continue
    circularity = 4 * np.pi * area / (perimeter ** 2)
    if circularity > 0.65:
        # Check if it has a reasonable size for a marked circle
        # In a 790x1456 image, the cells are around 70x70, so a circle radius should be 20-30, area 1200-2800.
        x, y, w, h = cv2.boundingRect(c)
        if 20 < w < 100 and 20 < h < 100:
            circular_contours.append((c, area, circularity, (x, y, w, h)))

print(f"Detected {len(circular_contours)} circular-like contours:")
for i, (c, area, circ, bbox) in enumerate(circular_contours[:30]):
    print(f"  Contour {i+1}: area={area:.1f}, circularity={circ:.2f}, bbox={bbox}")
