import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
puzzle_path = SCRIPT_DIR / "puzzles" / "puzzle_08.jpg"

img = cv2.imread(str(puzzle_path))
h, w = img.shape[:2]
half = h // 2
img_a = img[:half]

# Analyze column variance
gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
col_var = np.var(gray, axis=0)

# Print some statistics or find where the game panel starts
# Active game panel usually has higher variance and is surrounded by a white or solid border.
# Let's print out column variance values
print(f"Image width: {w}")
print("Column variances (first 100 columns):")
print(col_var[:100].round(1))
print("Column variances (last 100 columns):")
print(col_var[-100:].round(1))

# Let's find columns with very low variance or identify the boundary of the game area.
# In puzzle_08.jpg, there is a clean white frame around the game area.
# The white frame has very high intensity (near 255) and low variance.
col_mean = np.mean(gray, axis=0)
print("\nColumn means (first 100 columns):")
print(col_mean[:100].round(1))
print("Column means (last 100 columns):")
print(col_mean[-100:].round(1))
