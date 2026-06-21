import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
ans_path = SCRIPT_DIR / "correct_answers" / "answer_07.jpg"

img = cv2.imread(str(ans_path))
h, w = img.shape[:2]
half = h // 2
img_a = img[:half] # top panel where red circles are drawn

# Filter for the red color of the drawn circles
hsv = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, (0, 150, 150), (10, 255, 255))
mask2 = cv2.inRange(hsv, (170, 150, 150), (180, 255, 255))
mask = mask1 | mask2

# Find contours of red blobs
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_cnts = [c for c in cnts if cv2.contourArea(c) > 5]

print(f"Detected {len(valid_cnts)} red contours in top panel of answer_07.jpg:")
for i, c in enumerate(valid_cnts):
    (x, y), r = cv2.minEnclosingCircle(c)
    print(f"  Blob {i+1}: center=({int(x)},{int(y)}), radius={r:.1f}")
