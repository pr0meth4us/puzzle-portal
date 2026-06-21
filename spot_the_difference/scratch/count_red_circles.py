import cv2
import numpy as np
from pathlib import Path

def count_red(img_path):
    img = cv2.imread(str(img_path))
    if img is None: return 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Red has two ranges in HSV
    lower1 = np.array([0, 150, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 150, 50])
    upper2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = mask1 | mask2
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # filter tiny contours
    count = 0
    for c in contours:
        if cv2.contourArea(c) > 20: # arbitrary
            count += 1
    return count

for i in range(1, 9):
    p = f"correct_answers/answer_0{i}.jpg"
    print(f"answer_0{i}.jpg: {count_red(p)} red shapes")
