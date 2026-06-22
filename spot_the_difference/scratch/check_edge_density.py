import cv2
import sys
import numpy as np
from pathlib import Path

sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

puzzles = [
    ("puzzle_02.jpg", "validation_dataset"),
    ("puzzle_03.jpg", "validation_dataset"),
    ("puzzle_04.jpg", "validation_dataset"),
    ("puzzle_05.jpg", "validation_dataset"),
    ("puzzle_06.jpg", "validation_dataset"),
    ("puzzle_07.jpg", "puzzles"),
    ("puzzle_08.jpg", "puzzles"),
]

for puz, folder in puzzles:
    p_path = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference") / folder / puz
    combined = std.load_bgr(p_path)
    h, w = combined.shape[:2]
    # Slicing
    if puz == "puzzle_06.jpg":
        half = h // 2
        img_a = combined[:half]
    elif puz == "puzzle_04.jpg":
        img_a = combined
    else:
        cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
        img_a, img_b, _, _ = std.auto_slice(cropped_combined)
    
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    edges_a = cv2.Canny(gray_a, 50, 150)
    density = np.mean(edges_a == 255)
    print(f"{puz}: Edge density = {density:.4f}")
