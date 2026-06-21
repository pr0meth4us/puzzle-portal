import cv2
import numpy as np
from pathlib import Path
import spot_the_differences

SCRIPT_DIR = Path(__file__).resolve().parent
img = cv2.imread(str(SCRIPT_DIR / "validation_dataset/puzzle_03.jpg"))
h, w = img.shape[:2]

sift = cv2.SIFT_create()

# Test splits from 300 to 340
for sep in range(280, 361, 10):
    a = img[:sep, :]
    b = img[sep:, :]
    
    # Resize b to match a
    b_resized = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b_resized, cv2.COLOR_BGR2GRAY)
    
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)
    
    if des_a is None or des_b is None:
        print(f"sep={sep}: No descriptors")
        continue
        
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a, des_b, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            
    print(f"sep={sep}: good matches={len(good)}")
