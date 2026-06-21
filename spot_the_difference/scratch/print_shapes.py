import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
img = cv2.imread(str(SCRIPT_DIR / "validation_dataset/puzzle_03.jpg"))
print("Original shape:", img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
row_var = np.var(gray, axis=1)
content_rows = np.where(row_var > 50)[0]
print("Number of content rows with var > 50:", len(content_rows))
if len(content_rows) > 0:
    print(f"Content rows start: {content_rows[0]}, end: {content_rows[-1]}")
    
# Let's see with var > 10
content_rows_10 = np.where(row_var > 10)[0]
print("Number of content rows with var > 10:", len(content_rows_10))
if len(content_rows_10) > 0:
    print(f"Content rows (var > 10) start: {content_rows_10[0]}, end: {content_rows_10[-1]}")
