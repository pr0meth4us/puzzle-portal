import cv2
import sys
import numpy as np
from pathlib import Path

sys.path.append("/Users/nicksng/code/random/spot_the_difference")
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
    p_path = Path("/Users/nicksng/code/random/spot_the_difference") / folder / puz
    combined = std.load_bgr(p_path)
    h, w = combined.shape[:2]
    if puz == "puzzle_06.jpg":
        continue
    elif puz == "puzzle_04.jpg":
        img_a = combined
    else:
        cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
        img_a, img_b, _, _ = std.auto_slice(cropped_combined)
    
    gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    
    # Calculate local standard deviation in 5x5 neighborhood
    mean = cv2.blur(gray, (5, 5))
    mean_sq = cv2.blur(gray.astype(float)**2, (5, 5))
    variance = mean_sq - mean.astype(float)**2
    # Ensure no negative variance due to floating point precision
    variance = np.clip(variance, 0, None)
    std_dev = np.sqrt(variance)
    
    # Percentage of pixels with std_dev < 3.0 (flat/uniform regions)
    flat_pct = np.mean(std_dev < 3.0)
    print(f"{puz}: Flatness percentage = {flat_pct:.4f}")
