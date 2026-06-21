import cv2
import pytesseract
from PIL import Image
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
puzzle_path = SCRIPT_DIR / "puzzles" / "puzzle_07.jpg"

img = cv2.imread(str(puzzle_path))
h, w = img.shape[:2]
half = h // 2
img_a = img[:half]

for rot_name, rot_code in [("90_CW", cv2.ROTATE_90_CLOCKWISE), ("90_CCW", cv2.ROTATE_90_COUNTERCLOCKWISE)]:
    img_a_rot = cv2.rotate(img_a, rot_code)
    print(f"\n=== OCR on {rot_name} rotated panel ===")
    cfg = "-l eng+khm --psm 11"
    data = pytesseract.image_to_data(Image.fromarray(cv2.cvtColor(img_a_rot, cv2.COLOR_BGR2RGB)), config=cfg, output_type=pytesseract.Output.DICT)
    n_boxes = len(data.get('level', []))
    words = []
    for i in range(n_boxes):
        text = data['text'][i].strip()
        if text:
            rx, ry, rw, rh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            # Map rotated coordinates back to original panel coordinates
            if rot_code == cv2.ROTATE_90_CLOCKWISE:
                x = ry
                y = w - rx - rw
            else: # ROTATE_90_COUNTERCLOCKWISE
                x = h - ry - rh
                y = rx
            words.append((text, data['conf'][i], x, y, rw, rh))
            
    for w_info in words:
        if w_info[1] > 30 or "រូប" in w_info[0] or "ទី" in w_info[0]:
            print(f"  {w_info[0]} (conf={w_info[1]}): mapped_rect=({w_info[2]},{w_info[3]},{w_info[4]},{w_info[5]})")
