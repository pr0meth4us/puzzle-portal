import cv2
import numpy as np

# A script to carefully extract the number of red circles in the correct_answers
# We know the color is red or similar. Let's do it by finding connected components of non-image pixels?
# Actually, the user has drawn red/pink circles. Let's just do a wide color mask.

answers = {
    "answer_01.jpg": "puzzle_extra_05 vs 06",
    "answer_02.jpg": "puzzle_02.jpg",
    "answer_03.jpg": "puzzle_03.jpg",
    "answer_04.jpg": "puzzle_04.jpg (Swans)",
    "answer_05.jpg": "puzzle_05.jpg (Island)",
    "answer_06.jpg": "puzzle_06.jpg (Number grid)",
    "answer_07.jpg": "Waterfall 2?",
    "answer_08.jpg": "Bear?"
}

def count_circles(path):
    img = cv2.imread(path)
    if img is None: return -1
    
    # Try different color thresholds. Red, yellow, etc.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Let's just find anything that is heavily saturated and red/pink
    mask1 = cv2.inRange(hsv, (0, 100, 100), (20, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    
    # For puzzle_extra, the circles might be yellow?
    mask3 = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255)) # yellow
    # Green circles
    mask4 = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255)) # green
    mask = mask1 | mask2 | mask3 | mask4
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len([c for c in contours if cv2.contourArea(c) > 10])

for f, puz in answers.items():
    cnt = count_circles(f"correct_answers/{f}")
    print(f"{f} ({puz}): {cnt} shapes -> {cnt//2} diffs?")
