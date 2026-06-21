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

# Crop and resize both to a standard size (e.g. 360 width, 400 height)
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
        x1, y1 = c * cell_w, r * cell_h
        x2, y2 = (c + 1) * cell_w, (r + 1) * cell_h
        
        cell_a = gray_a[y1:y2, x1:x2]
        cell_b = gray_b[y1:y2, x1:x2]
        
        # Absolute difference
        diff_img = cv2.absdiff(cell_a, cell_b)
        mean_diff = np.mean(diff_img)
        
        diffs.append((r, c, mean_diff))

# Sort diffs by mean_diff to see the distribution
diffs.sort(key=lambda x: x[2], reverse=True)

print("Top 30 cell differences:")
for i, (r, c, score) in enumerate(diffs[:30]):
    print(f"Rank {i+1}: Row {r+1}, Col {c+1} -> score={score:.2f}")

# Let's count how many are above a threshold.
# Let's look at the scores and find a good threshold.
