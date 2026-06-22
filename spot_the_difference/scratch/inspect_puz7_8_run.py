import cv2
import numpy as np

def inspect_puzzle(puz_path, ans_path, name):
    puz = cv2.imread(puz_path)
    ans = cv2.imread(ans_path)
    if puz is None or ans is None:
        print(f"Error loading {puz_path} or {ans_path}")
        return
    print(f"\n==================== {name} ====================")
    print(f"  Puzzle shape: {puz.shape}")
    print(f"  Answer shape: {ans.shape}")
    
    hsv = cv2.cvtColor(ans, cv2.COLOR_BGR2HSV)
    
    # Red has hue near 0 and 180. Let's use looser bounds to capture all hand-drawn red circles.
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([168, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2
    
    cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dots = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 10:  # ignore tiny noise pixels
            (cx, cy), r = cv2.minEnclosingCircle(c)
            # Make sure it's not a border watermark or banner area if applicable
            # (we want to check if they form circles of radius between 5 and 30)
            if 3 < r < 40:
                dots.append((cx, cy, r, area))
            
    filtered_dots = []
    for cx, cy, r, area in dots:
        too_close = False
        for kx, ky, _, _ in filtered_dots:
            if np.hypot(cx - kx, cy - ky) < 15:
                too_close = True
                break
        if not too_close:
            filtered_dots.append((cx, cy, r, area))
            
    filtered_dots.sort(key=lambda x: x[1])
    print(f"  Unique red dots found in Answer: {len(filtered_dots)}")
    for i, (cx, cy, r, area) in enumerate(filtered_dots):
        print(f"    Dot {i+1}: Center=({cx:.1f}, {cy:.1f}), r={r:.1f}, Area={area:.1f}")

inspect_puzzle("puzzles/puzzle_07.jpg", "correct_answers/answer_07.jpg", "Puzzle 7")
inspect_puzzle("puzzles/puzzle_08.jpg", "correct_answers/answer_08.jpg", "Puzzle 8")
