import sys
from pathlib import Path
import cv2
import numpy as np

SCRIPT_DIR = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference")
sys.path.append(str(SCRIPT_DIR.parent))
import spot_the_difference.spot_the_differences as std

img_a_path = SCRIPT_DIR / "puzzles" / "puzzle_extra_05.jpg"
img_b_path = SCRIPT_DIR / "puzzles" / "puzzle_extra_06.jpg"

img_a = std.load_bgr(str(img_a_path))
img_b = std.load_bgr(str(img_b_path))

print("Aligning...")
img_b_aligned, valid_y_range, H_align, valid_mask = std.align(img_a, img_b, skip_ecc=False)

print("Detecting...")
circles, count = std.detect_line(img_a, img_b_aligned)

H_inv = np.linalg.inv(H_align) if H_align is not None else None
h, w = img_a.shape[:2]
diff = cv2.absdiff(img_a, img_b_aligned)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

bmask = np.ones((h, w), dtype=np.uint8) * 255
bmask[:12, :] = 0
bmask[h-12:, :] = 0
bmask[:, :12] = 0
bmask[:, w-12:] = 0
if valid_mask is not None:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_valid = cv2.erode(valid_mask, kernel)
    bmask = cv2.bitwise_and(bmask, eroded_valid)

gray_diff = cv2.bitwise_and(gray_diff, bmask)

# Warped circles for fallback
pw, ph = img_b.shape[1], img_b.shape[0]
if H_inv is not None:
    pts = np.float32([[cx, cy] for cx, cy, r in circles]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
    warped_circles = [(int(np.clip(mx, 0, pw-1)), int(np.clip(my, 0, ph-1)), r)
                      for (mx, my), (_, _, r) in zip(mapped, circles)]
else:
    warped_circles = [(cx, cy, r) for cx, cy, r in circles]

for idx, (cx, cy, r) in enumerate(circles):
    cx, cy = int(cx), int(cy)
    r_size = max(40, int(r * 1.5))
    y1, y2 = max(0, cy - r_size), min(h, cy + r_size)
    x1, x2 = max(0, cx - r_size), min(w, cx + r_size)
    roi = gray_diff[y1:y2, x1:x2].copy()
    roi[0:2, :] = 0
    roi[-2:, :] = 0
    roi[:, 0:2] = 0
    roi[:, -2:] = 0
    max_val = np.max(roi)
    thresh_val = max(8, int(max_val * 0.25))
    _, th = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_to_draw = []
    if cnts:
        for c in cnts:
            if cv2.contourArea(c) >= 4:
                contours_to_draw.append(c)
                
    if contours_to_draw:
        print(f"Difference {idx+1}: CONTOURS (count={len(contours_to_draw)})")
    else:
        print(f"Difference {idx+1}: CIRCLE")
