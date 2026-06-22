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
    if puz == "puzzle_06.jpg":
        continue
    elif puz == "puzzle_04.jpg":
        # Swan mode has multiple sub-panels, let's skip for simple homography check
        continue
    else:
        cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
        img_a, img_b, _, _ = std.auto_slice(cropped_combined)
    
    img_b_aligned, valid_y_range, H = std.align(img_a, img_b, skip_ecc=False)
    if H is not None:
        # Check scale/rotation components
        # H is a 3x3 matrix.
        # H[0,0] and H[1,1] are cos(theta)*scale, H[0,1] and H[1,0] are sin(theta)*scale, etc.
        # Let's print H and trace/determinant/deviation from pure translation
        # Pure translation has H[0,0]=1, H[0,1]=0, H[1,0]=0, H[1,1]=1, H[2,0]=0, H[2,1]=0
        dev_rot_scale = np.abs(H[0,0] - 1.0) + np.abs(H[1,1] - 1.0) + np.abs(H[0,1]) + np.abs(H[1,0]) + np.abs(H[2,0]) + np.abs(H[2,1])
        print(f"{puz}: Homography H:\n{H}\nDeviation from pure translation: {dev_rot_scale:.6f}\n")
    else:
        print(f"{puz}: Homography H is None\n")
