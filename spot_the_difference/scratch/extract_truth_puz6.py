import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ans_img = cv2.imread(str(SCRIPT_DIR / "correct_answers/answer_06.jpg"))
puz_img = cv2.imread(str(SCRIPT_DIR / "validation_dataset/puzzle_06.jpg"))

h_ans, w_ans = ans_img.shape[:2]
h_puz, w_puz = puz_img.shape[:2]

print(f"Answer shape: {ans_img.shape}")
print(f"Puzzle shape: {puz_img.shape}")

# Find green mask in answer
hsv = cv2.cvtColor(ans_img, cv2.COLOR_BGR2HSV)
# Hue 35 to 95 is green
mask = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))

# Filter mask to remove small noise
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
circles = []
for c in cnts:
    area = cv2.contourArea(c)
    if 50 < area < 5000:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        circles.append((int(cx), int(cy), int(r)))

print(f"Detected {len(circles)} green circles in answer:")

# Since answer_06.jpg is of size 790x1456, let's see if it has two panels.
# Let's segment the grid on the answer image if possible.
# Let's crop the bottom part of answer_06.jpg (where B panel is)
# SIFT mapped answer_06.jpg to puzzle_06.jpg.
# Let's find the grid bbox in the answer image!
gray = cv2.cvtColor(ans_img, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
if np.mean(binary) < 128:
    binary = cv2.bitwise_not(binary)
cnts_grid, _ = cv2.findContours(cv2.bitwise_not(binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find bboxes of large rectangles that could be grids
grid_bboxes = []
for c in cnts_grid:
    x, y, w_box, h_box = cv2.boundingRect(c)
    if w_box > w_ans * 0.4 and h_box > h_ans * 0.2:
        grid_bboxes.append((x, y, w_box, h_box))

print("Detected grid-like bboxes in answer_06.jpg:")
for i, bbox in enumerate(grid_bboxes):
    print(f"  Grid {i+1}: {bbox}")

# If we have green circles, let's see which grid they belong to.
# Typically they are drawn on the bottom panel grid.
# Let's map each circle to a row, col in the bottom grid.
if grid_bboxes:
    # Sort grids by Y coordinate (top to bottom)
    grid_bboxes.sort(key=lambda x: x[1])
    # The bottom grid is the last one
    gx, gy, gw, gh = grid_bboxes[-1]
    print(f"Using bottom grid bbox: x={gx}, y={gy}, w={gw}, h={gh}")
    
    rows, cols = 10, 9
    cell_w = gw / cols
    cell_h = gh / rows
    
    truth_cells = []
    for cx, cy, r in circles:
        if gx <= cx <= gx+gw and gy <= cy <= gy+gh:
            col = int((cx - gx) / cell_w) + 1
            row = int((cy - gy) / cell_h) + 1
            truth_cells.append((row, col))
            
    truth_cells.sort()
    print(f"\nGround Truth Diff Cells (count: {len(truth_cells)}):")
    for r, c in truth_cells:
        print(f"  Row {r}, Col {c}")
