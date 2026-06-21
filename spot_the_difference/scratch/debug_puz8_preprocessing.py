import cv2
import pytesseract
from PIL import Image
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
crop_path = SCRIPT_DIR / "scratch" / "crop_top_a.png"

img = cv2.imread(str(crop_path))

# Let's try 2x upscaling
img_2x = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)

# Let's try binarization on 2x upscaled
gray = cv2.cvtColor(img_2x, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)

# Run Tesseract with eng+khm on original, 2x, and 2x binary
for name, processed in [("original", img), ("2x", img_2x), ("2x_binary", binary)]:
    print(f"\n--- Preprocessing: {name} ---")
    for psm in [3, 11, 12]:
        cfg = f"-l eng+khm --psm {psm}"
        data = pytesseract.image_to_data(Image.fromarray(processed), config=cfg, output_type=pytesseract.Output.DICT)
        n_boxes = len(data.get('level', []))
        words = []
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if text:
                words.append((text, data['conf'][i]))
        if words:
            print(f"  PSM {psm}: {words}")
        else:
            print(f"  PSM {psm}: None")
