import cv2
import numpy as np

img = cv2.imread("puzzles/puzzle_07.jpg")
if img is None:
    print("Could not read image.")
    exit()

h, w = img.shape[:2]
panel_h = 1006
panel_b = img[panel_h:, :]

# Print the mean of each row from y=900 to 1006 in panel B
print("Panel B bottom row means:")
for y in range(900, panel_h, 10):
    row = panel_b[y, :]
    print(f"  y={y}: mean BGR={np.mean(row, axis=0)}")
