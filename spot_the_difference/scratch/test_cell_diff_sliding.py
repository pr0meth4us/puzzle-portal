import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
img = cv2.imread(str(SCRIPT_DIR / "validation_dataset/puzzle_06.jpg"))
h, w = img.shape[:2]
sep = h // 2
img_a, img_b = img[:sep], img[sep:]

def get_grid_bbox(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
    if np.mean(binary) < 128:
        binary = cv2.bitwise_not(binary)
    cnts, _ = cv2.findContours(cv2.bitwise_not(binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        if cw > img_bgr.shape[1] * 0.7 and ch > img_bgr.shape[0] * 0.7:
            return x, y, cw, ch
    return 0, 0, img_bgr.shape[1], img_bgr.shape[0]

xa, ya, wa, ha = get_grid_bbox(img_a)
xb, yb, wb, hb = get_grid_bbox(img_b)

target_w, target_h = 360, 400
grid_a_crop = cv2.resize(img_a[ya:ya+ha, xa:xa+wa], (target_w, target_h))
grid_b_crop = cv2.resize(img_b[yb:yb+hb, xb:xb+wb], (target_w, target_h))

gray_a = cv2.cvtColor(grid_a_crop, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(grid_b_crop, cv2.COLOR_BGR2GRAY)

rows, cols = 10, 9
cell_w = target_w // cols
cell_h = target_h // rows

diffs = []
for r in range(rows):
    for c in range(cols):
        # Coordinates of cell in A
        x1_a = int(xa + c * (wa / cols))
        y1_a = int(ya + r * (ha / rows))
        x2_a = int(xa + (c + 1) * (wa / cols))
        y2_a = int(ya + (r + 1) * (ha / rows))
        
        # Coordinates of cell in B
        x1_b = int(xb + c * (wb / cols))
        y1_b = int(yb + r * (hb / rows))
        x2_b = int(xb + (c + 1) * (wb / cols))
        y2_b = int(yb + (r + 1) * (hb / rows))
        
        # Exclude cell border by shrinking the crop window (e.g. crop internal 70%)
        margin_x = int((x2_a - x1_a) * 0.15)
        margin_y = int((y2_a - y1_a) * 0.15)
        
        inner_a = gray_a = cv2.cvtColor(img_a[y1_a+margin_y : y2_a-margin_y, x1_a+margin_x : x2_a-margin_x], cv2.COLOR_BGR2GRAY)
        
        # Find minimum difference over small shifts in B
        min_score = float('inf')
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y1_b_shift = max(0, y1_b + margin_y + dy)
                y2_b_shift = min(img_b.shape[0], y2_b - margin_y + dy)
                x1_b_shift = max(0, x1_b + margin_x + dx)
                x2_b_shift = min(img_b.shape[1], x2_b - margin_x + dx)
                
                # Check height and width match inner_a
                inner_b = cv2.cvtColor(img_b[y1_b_shift:y2_b_shift, x1_b_shift:x2_b_shift], cv2.COLOR_BGR2GRAY)
                if inner_b.shape != inner_a.shape:
                    inner_b = cv2.resize(inner_b, (inner_a.shape[1], inner_a.shape[0]))
                    
                score = np.mean(cv2.absdiff(inner_a, inner_b))
                if score < min_score:
                    min_score = score
                    
        diffs.append((r, c, min_score))

diffs.sort(key=lambda x: x[2], reverse=True)

print("Top 30 cell differences (with shift & margin):")
for i, (r, c, score) in enumerate(diffs[:30]):
    print(f"Rank {i+1}: Row {r+1}, Col {c+1} -> score={score:.2f}")
