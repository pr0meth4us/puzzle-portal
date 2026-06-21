import cv2
import pytesseract
from PIL import Image
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
puzzle_path = SCRIPT_DIR / "puzzles" / "puzzle_08.jpg"

img = cv2.imread(str(puzzle_path))
h, w = img.shape[:2]
half = h // 2
img_b = img[half:]

print("Running full image OCR with PSM 11...")
cfg = "-l eng+khm --psm 11"
data = pytesseract.image_to_data(Image.fromarray(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)), config=cfg, output_type=pytesseract.Output.DICT)
n_boxes = len(data.get('level', []))
words = []
for i in range(n_boxes):
    text = data['text'][i].strip()
    if text:
        words.append((text, data['conf'][i], data['left'][i], data['top'][i], data['width'][i], data['height'][i]))

print(f"Detected {len(words)} words:")
for w_info in words:
    print(f"  {w_info[0]} (conf={w_info[1]}): rect=({w_info[2]},{w_info[3]},{w_info[4]},{w_info[5]})")
