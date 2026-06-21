import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
img = cv2.imread(str(SCRIPT_DIR / "validation_dataset/puzzle_06.jpg"))

# Slicing
h, w = img.shape[:2]
sep = h // 2
img_a, img_b = img[:sep], img[sep:]

def debug_ocr(img_bgr, name):
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray   = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 8)
    if np.mean(binary) < 128:
        binary = cv2.bitwise_not(binary)

    # Let's crop grid content
    PAD = 4
    cnts, _ = cv2.findContours(cv2.bitwise_not(binary),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        print(f"[{name}] bounding rect: {x}, {y}, {cw}, {ch} (img w={w}, h={h//2})")
        if cw > w * 0.7 and ch > (h//2) * 0.7:
            x1 = max(0,   x  + PAD)
            y1 = max(0,   y  + PAD)
            x2 = min(w,   x  + cw - PAD)
            y2 = min(h//2,   y  + ch - PAD)
            cropped = binary[y1:y2, x1:x2]
            print(f"[{name}] Crop: ({x1},{y1}) -> ({x2},{y2})")
        else:
            print(f"[{name}] Bounding rect not large enough")
            cropped = binary[PAD:h//2-PAD, PAD:w-PAD]
    else:
        cropped = binary[PAD:h//2-PAD, PAD:w-PAD]

    cfg  = "--psm 6 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(Image.fromarray(cropped), config=cfg).strip()
    print(f"[{name}] Raw OCR text:\n{text}\n")

print("DEBUG OCR FOR PUZZLE 6:")
debug_ocr(img_a, "img_a")
debug_ocr(img_b, "img_b")
