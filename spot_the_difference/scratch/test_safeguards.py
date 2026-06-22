import cv2
import numpy as np
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

img_a = std.load_bgr("spot_the_difference/puzzles/puzzle_extra_05.jpg")
img_b = std.load_bgr("spot_the_difference/puzzles/puzzle_extra_06.jpg")

img_b_aligned, valid_y_range, H_align, valid_mask = std.align(img_a, img_b, skip_ecc=False)

h, w  = img_a.shape[:2]
img_b_aligned = cv2.resize(img_b_aligned, (w, h), interpolation=cv2.INTER_LANCZOS4)
score, diff = std.ssim(std._gray(img_a), std._gray(img_b_aligned), full=True)
inv      = cv2.bitwise_not((diff * 255).clip(0, 255).astype(np.uint8))
_, thresh = cv2.threshold(inv, 30, 255, cv2.THRESH_BINARY)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k)

border = max(8, int(min(h, w) * 0.02))
bmask  = np.zeros_like(thresh)
bmask[border:h - border, border:w - border] = 255

# Apply valid_mask
if valid_mask is not None:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_valid = cv2.erode(valid_mask, kernel)
    bmask = cv2.bitwise_and(bmask, eroded_valid)

# Apply margins (top=35, bottom=12, left=20, right=20)
bmask[:35, :] = 0
bmask[h-12:, :] = 0
bmask[:, :20] = 0
bmask[:, w-20:] = 0

# Modified is_watermark_text that excludes sers and llge
def custom_is_watermark_text(text: str) -> bool:
    t_lower = text.lower()
    eng_keywords = [
        "enterprises", "digital", "gov", "kh", "copyright", "john", "©", 
        "c0pyright", "ste", "bae", "col", "jus"
    ]
    khm_keywords = ["រូប", "រប", "ទី", "ខុស", "គ្នា", "ស្វែង", "រក", "ចំណុច", "រូបភាព", "របភាព"]
    for k in khm_keywords:
        if k in text:
            return True
    for k in eng_keywords:
        if k in t_lower:
            return True
    return False

def custom_mask_ocr_text(img_a, img_b, bmask):
    import pytesseract
    from PIL import Image
    h_panel, w_panel = img_a.shape[:2]
    for img in (img_a, img_b):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        data = pytesseract.image_to_data(pil_img, config="-l eng+khm --psm 11", output_type=pytesseract.Output.DICT)
        n_boxes = len(data.get('level', []))
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if text:
                if not custom_is_watermark_text(text):
                    continue
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                if w > bmask.shape[1] * 0.35 or h > bmask.shape[0] * 0.25:
                    continue
                    
                pad = 30
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(bmask.shape[1], x + w + pad)
                y2 = min(bmask.shape[0], y + h + pad)
                bmask[y1:y2, x1:x2] = 0

custom_mask_ocr_text(img_a, img_b_aligned, bmask)

thresh_masked = cv2.bitwise_and(thresh, bmask)

contours, _ = cv2.findContours(thresh_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cdiff = np.mean(cv2.absdiff(img_a, img_b_aligned).astype(np.float32), axis=2)
ga    = std._gray(img_a).astype(np.float64)
gb    = std._gray(img_b_aligned).astype(np.float64)

candidates = []
for cnt in contours:
    if cv2.contourArea(cnt) < 5:
        continue
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
    delta = cv2.mean(cdiff, mask=mask)[0]
    if delta < 5.0:
        continue
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    cx, cy, r   = int(cx), int(cy), int(r)
    x1 = max(0, cx-r-5); y1 = max(0, cy-r-5)
    x2 = min(w, cx+r+5); y2 = min(h, cy+r+5)
    peak = float(np.abs(ga[y1:y2, x1:x2] - gb[y1:y2, x1:x2]).max())
    if peak < max(80, int(min(h, w) * 0.45)):
        continue
    max_r = int(min(h, w) * 0.15)
    candidates.append((cx, cy, min(max(r + 15, 20), max_r), delta))

candidates.sort(key=lambda x: -x[3])
kept = []
LINE_NMS_RADIUS = max(20, int(min(h, w) * 0.12))
for cx, cy, r, d in candidates:
    suppressed = False
    for kx, ky, _, kd in kept:
        dist = ((cx-kx)**2+(cy-ky)**2)**0.5
        if dist < LINE_NMS_RADIUS:
            suppressed = True
            break
    if not suppressed:
        kept.append((cx, cy, r, d))

print(f"Detections ({len(kept)} total):")
for idx, (cx, cy, r, d) in enumerate(kept):
    print(f"  Circle {idx+1}: center=({cx}, {cy}), r={r}, delta={d:.1f}")

gt_mapped = [
    (109.4, 64.1),
    (544.9, 84.8),
    (294.0, 87.6),
    (438.8, 100.0),
    (112.0, 127.5),
    (207.1, 218.5),
    (557.4, 291.7),
    (55.2, 313.8),
    (321.5, 364.9),
    (517.5, 395.2)
]

print("\nMatching to GT:")
matched = set()
for idx, (cx, cy, r, d) in enumerate(kept):
    closest_dist = float('inf')
    closest_gt = -1
    for i, (gcx, gcy) in enumerate(gt_mapped):
        dist = np.sqrt((cx - gcx)**2 + (cy - gcy)**2)
        if dist < closest_dist:
            closest_dist = dist
            closest_gt = i
    if closest_dist < 60:
        matched.add(closest_gt)
        print(f"  Detection {idx+1} at ({cx}, {cy}) -> MATCHED GT {closest_gt+1} (dist={closest_dist:.1f})")
    else:
        print(f"  Detection {idx+1} at ({cx}, {cy}) -> FALSE POSITIVE (closest GT {closest_gt+1} dist={closest_dist:.1f})")

print(f"\nTotal matched GTs: {len(matched)} / {len(gt_mapped)}")
print("Missed GTs:")
for i in range(len(gt_mapped)):
    if i not in matched:
        print(f"  GT {i+1} at {gt_mapped[i]}")
