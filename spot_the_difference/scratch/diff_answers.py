import cv2
import numpy as np

def find_marked_circles_by_diff(puz_path, ans_path, name):
    puz = cv2.imread(puz_path)
    ans = cv2.imread(ans_path)
    if puz is None or ans is None:
        print(f"Error loading {puz_path} or {ans_path}")
        return
        
    if puz.shape != ans.shape:
        ans = cv2.resize(ans, (puz.shape[1], puz.shape[0]))
        
    diff = cv2.absdiff(puz, ans)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 15:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w/2, y + h/2
            circles.append((cx, cy, w, h, area))
            
    grouped = []
    for cx, cy, w, h, area in circles:
        too_close = False
        for g in grouped:
            if np.hypot(cx - g[0], cy - g[1]) < 25:
                too_close = True
                break
        if not too_close:
            grouped.append((cx, cy, w, h, area))
            
    print(f"--- {name} ---")
    for i, (cx, cy, w, h, area) in enumerate(grouped):
        print(f"  Diff {i+1}: Center=({cx:.1f}, {cy:.1f}), Size={w}x{h}, Area={area:.0f}")

find_marked_circles_by_diff("validation_dataset/puzzle_03.jpg", "correct_answers/answer_03.jpg", "Puzzle 3")
