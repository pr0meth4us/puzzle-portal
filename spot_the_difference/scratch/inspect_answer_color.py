import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
img = cv2.imread(str(SCRIPT_DIR / "correct_answers/answer_06.jpg"))
if img is None:
    print("Could not read answer_06.jpg")
    exit()

h, w = img.shape[:2]
print(f"answer_06.jpg size: {w}x{h}")

# Let's count non-white, non-black, non-gray pixels
# Let's count bright, saturated pixels
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
s = hsv[:, :, 1]
v = hsv[:, :, 2]

saturated_coords = np.argwhere((s > 50) & (v > 50))
print(f"Number of saturated pixels: {len(saturated_coords)}")
if len(saturated_coords) > 0:
    print("Sample saturated pixels (Y, X, H, S, V, BGR):")
    for y, x in saturated_coords[:10]:
        print(f"({y}, {x}) -> HSV={hsv[y, x].tolist()}, BGR={img[y, x].tolist()}")
