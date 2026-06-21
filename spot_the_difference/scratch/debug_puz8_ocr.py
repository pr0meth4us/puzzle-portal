import cv2
import pytesseract
from PIL import Image
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
crop_path = SCRIPT_DIR / "scratch" / "crop_top_a.png"

img = cv2.imread(str(crop_path))
h, w = img.shape[:2]

# Crop tightly around the center text
# The text is centered horizontally: w // 2 is 576. Let's crop x from 500 to 650, y from 0 to 60.
tight_crop = img[0:80, 500:650]
cv2.imwrite(str(SCRIPT_DIR / "scratch" / "crop_tight.png"), tight_crop)

print("Tight crop saved. Running OCR...")
for lang in ["khm", "eng+khm"]:
    for psm in [3, 6, 7, 8, 11]:
        cfg = f"-l {lang} --psm {psm}"
        text = pytesseract.image_to_string(Image.fromarray(cv2.cvtColor(tight_crop, cv2.COLOR_BGR2RGB)), config=cfg).strip()
        if text:
            print(f"  lang={lang} PSM {psm}: '{text}'")
