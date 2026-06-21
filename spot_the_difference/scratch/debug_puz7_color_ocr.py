import cv2
import pytesseract
from PIL import Image
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent
puzzle_path = SCRIPT_DIR / "puzzles" / "puzzle_07.jpg"

img = cv2.imread(str(puzzle_path))
h, w = img.shape[:2]
half = h // 2
img_a = img[:half]
img_b = img[half:]

# Crop to rightmost 150 pixels
crop_a = img_a[:, w - 150:]
crop_b = img_b[:, w - 150:]

# Convert to HSV
hsv_a = cv2.cvtColor(crop_a, cv2.COLOR_BGR2HSV)
hsv_b = cv2.cvtColor(crop_b, cv2.COLOR_BGR2HSV)

# Yellow and Red masks on the crops
mask_yellow = cv2.inRange(hsv_a, (15, 80, 80), (35, 255, 255))
mask_red1 = cv2.inRange(hsv_b, (0, 80, 80), (10, 255, 255))
mask_red2 = cv2.inRange(hsv_b, (170, 80, 80), (180, 255, 255))
mask_red = mask_red1 | mask_red2

# Rotate CCW to make vertical text horizontal
rot_yellow = cv2.rotate(mask_yellow, cv2.ROTATE_90_COUNTERCLOCKWISE)
rot_red = cv2.rotate(mask_red, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imwrite(str(SCRIPT_DIR / "scratch" / "crop_rot_yellow.png"), rot_yellow)
cv2.imwrite(str(SCRIPT_DIR / "scratch" / "crop_rot_red.png"), rot_red)

# Run Tesseract
print("Running OCR on cropped, rotated color masks...")
for name, processed in [("Yellow (CCW)", rot_yellow), ("Red (CCW)", rot_red)]:
    cfg = "-l eng+khm --psm 11"
    text = pytesseract.image_to_string(Image.fromarray(processed), config=cfg).strip()
    print(f"  {name}: '{text}'")
