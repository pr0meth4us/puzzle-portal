import cv2
import numpy as np
import glob
import os

for path in sorted(glob.glob("spot_the_difference/correct_answers/answer_*.jpg")):
    img = cv2.imread(path)
    if img is None:
        continue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total = 0
    circles = 0
    contours = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 10:
            continue
        total += 1
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (peri ** 2) if peri > 0 else 0
        if circ >= 0.65:
            circles += 1
        else:
            contours += 1
            
    print(f"{os.path.basename(path)} (shape {img.shape}): total_red={total}, circles={circles}, contours={contours}")
