import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
img = cv2.imread(str(SCRIPT_DIR / "correct_answers/answer_06.jpg"))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
s = hsv[:, :, 1]
v = hsv[:, :, 2]

# Segment by hue ranges
ranges = {
    "Red/Orange/Yellow (0-30)": ((0, 50, 50), (30, 255, 255)),
    "Green (30-90)": ((30, 50, 50), (90, 255, 255)),
    "Blue/Cyan (90-150)": ((90, 50, 50), (150, 255, 255)),
    "Red/Magenta (150-180)": ((150, 50, 50), (180, 255, 255))
}

for name, (low, high) in ranges.items():
    mask = cv2.inRange(hsv, low, high)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_cnts = [c for c in cnts if 20 < cv2.contourArea(c) < 10000]
    print(f"{name}: {np.sum(mask > 0)} pixels matching, {len(valid_cnts)} valid contours.")
