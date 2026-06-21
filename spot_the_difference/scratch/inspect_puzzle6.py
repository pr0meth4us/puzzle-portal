import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
img = cv2.imread(str(SCRIPT_DIR / "validation_dataset/puzzle_06.jpg"))
h, w = img.shape[:2]
sep = h // 2
img_a, img_b = img[:sep], img[sep:]

def inspect_grid(panel, name):
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    # Thresholding
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # Save thresholded image to see if lines or text are clear
    cv2.imwrite(str(SCRIPT_DIR / f"results/inspect_{name}_thresh.png"), thresh)
    
    # Run HoughLines to detect horizontal/vertical lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    print(f"[{name}] Detected {len(lines) if lines is not None else 0} line segments.")

print("Inspecting puzzle_06...")
inspect_grid(img_a, "img_a")
inspect_grid(img_b, "img_b")
