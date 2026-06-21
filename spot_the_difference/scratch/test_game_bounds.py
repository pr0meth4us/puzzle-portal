import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent

def find_game_bounds(gray_img):
    h, w = gray_img.shape
    col_var = np.var(gray_img, axis=0)
    col_mean = np.mean(gray_img, axis=0)
    
    left_bound = 0
    right_bound = w
    
    # Scan from left to find the white border band
    white_cols_l = []
    for x in range(w // 3):
        if col_mean[x] > 220 and col_var[x] < 500:
            white_cols_l.append(x)
    if white_cols_l:
        left_bound = max(white_cols_l) + 2
        
    # Scan from right to find the white border band
    white_cols_r = []
    for x in range(w - 1, 2 * w // 3, -1):
        if col_mean[x] > 220 and col_var[x] < 500:
            white_cols_r.append(x)
    if white_cols_r:
        right_bound = min(white_cols_r) - 2
        
    return left_bound, right_bound

for puz_name in ["puzzle_07.jpg", "puzzle_08.jpg"]:
    p_path = SCRIPT_DIR / "puzzles" / puz_name
    img = cv2.imread(str(p_path))
    if img is None: continue
    h, w = img.shape[:2]
    half = h // 2
    img_a = img[:half]
    gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    l_b, r_b = find_game_bounds(gray)
    print(f"{puz_name}: width={w}, detected bounds=({l_b}, {r_b})")
