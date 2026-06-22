import cv2
import numpy as np

ans = cv2.imread("correct_answers/answer_07.jpg")
h, w = ans.shape[:2]
half = h // 2

top_panel = ans[:half]
bottom_panel = ans[half:]

def find_red(img, name):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 10:
            (cx, cy), r = cv2.minEnclosingCircle(c)
            # Filter by radius to match drawn circles (usually 10 to 30 pixels in 480-width image)
            if 8 < r < 35:
                dots.append((cx, cy, r))
                
    # Filter close dots
    filtered = []
    for cx, cy, r in dots:
        if not any(np.hypot(cx - kx, cy - ky) < 15 for kx, ky, _ in filtered):
            filtered.append((cx, cy, r))
            
    filtered.sort(key=lambda x: x[1])
    print(f"\n{name} red circles (count={len(filtered)}):")
    for i, (cx, cy, r) in enumerate(filtered):
        print(f"  {i+1}: ({cx:.1f}, {cy:.1f}), r={r:.1f}")

find_red(top_panel, "Top Panel")
find_red(bottom_panel, "Bottom Panel")
