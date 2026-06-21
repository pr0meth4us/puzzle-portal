import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
img = cv2.imread(str(SCRIPT_DIR / "validation_dataset/puzzle_06.jpg"))
h, w = img.shape[:2]
sep = h // 2
img_a, img_b = img[:sep], img[sep:]

print(f"img_a shape: {img_a.shape}")
print(f"img_b shape: {img_b.shape}")

def get_grid_bbox_debug(img_bgr, name):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
    if np.mean(binary) < 128:
        binary = cv2.bitwise_not(binary)
    cnts, _ = cv2.findContours(cv2.bitwise_not(binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        print(f"[{name}] largest contour area: {cv2.contourArea(largest)}, bbox: x={x}, y={y}, w={cw}, h={ch}")
        return x, y, cw, ch
    return 0, 0, img_bgr.shape[1], img_bgr.shape[0]

get_grid_bbox_debug(img_a, "A")
get_grid_bbox_debug(img_b, "B")
