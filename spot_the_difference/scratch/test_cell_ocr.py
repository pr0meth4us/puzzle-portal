import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
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

print(f"Grid A bbox: x={xa}, y={ya}, w={wa}, h={ha}")
print(f"Grid B bbox: x={xb}, y={yb}, w={wb}, h={hb}")

rows, cols = 10, 9

def ocr_cell(img_bgr, gx, gy, gw, gh, r, c):
    cell_w = gw / cols
    cell_h = gh / rows
    
    # Calculate cell coordinates
    x1 = int(gx + c * cell_w)
    y1 = int(gy + r * cell_h)
    x2 = int(gx + (c + 1) * cell_w)
    y2 = int(gy + (r + 1) * cell_h)
    
    # Crop the cell
    cell = img_bgr[y1:y2, x1:x2]
    
    # Preprocess cell: convert to grayscale, resize for better OCR, threshold
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (0,0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Run OCR with single character PSM (10) or single word/line PSM (8/7)
    cfg = "--psm 10 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(thresh, config=cfg).strip()
    # Filter non-digits
    digits = re.findall(r'\d', text)
    return digits[0] if digits else "?"

grid_a = []
for r in range(rows):
    row = []
    for c in range(cols):
        val = ocr_cell(img_a, xa, ya, wa, ha, r, c)
        row.append(val)
    grid_a.append(row)

grid_b = []
for r in range(rows):
    row = []
    for c in range(cols):
        val = ocr_cell(img_b, xb, yb, wb, hb, r, c)
        row.append(val)
    grid_b.append(row)

print("\nGrid A:")
for row in grid_a:
    print(" ".join(row))

print("\nGrid B:")
for row in grid_b:
    print(" ".join(row))

diffs = 0
for r in range(rows):
    for c in range(cols):
        if grid_a[r][c] != grid_b[r][c]:
            diffs += 1
            print(f"Diff at row={r+1} col={c+1}: top={grid_a[r][c]} bottom={grid_b[r][c]}")

print(f"\nTotal discrepancies: {diffs}")
