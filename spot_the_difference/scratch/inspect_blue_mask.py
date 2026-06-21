import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
img = cv2.imread(str(SCRIPT_DIR / "correct_answers/answer_06.jpg"))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (90, 50, 50), (150, 255, 255))
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Blue/cyan contours count: {len(cnts)}")
for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    print(f"  Contour {i+1}: area={area}, bbox=({x},{y},{w},{h})")
