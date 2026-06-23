import cv2
import numpy as np
import sys
sys.path.append(".")
import engine_v3

img_a = engine_v3.load_bgr("puzzles/puzzle_extra_05.jpg")
img_b = engine_v3.load_bgr("puzzles/puzzle_extra_06.jpg")

img_b_aligned, H_align, valid_mask = engine_v3.align(img_a, img_b)

h, w = img_a.shape[:2]
img_b_resized = cv2.resize(img_b_aligned, (w, h), interpolation=cv2.INTER_LANCZOS4)

min_area   = engine_v3._min_area(w, h)
merge_r    = max(20, int(min(h, w) * 0.10))
nms_r      = merge_r
max_r      = int(min(h, w) * 0.15)
c_pad      = engine_v3._circle_pad(w)
border_px  = max(8, int(min(h, w) * 0.02))
ssim_thresh = 30
morph_k     = 3
peak_min    = 120

score, diff = engine_v3.ssim(engine_v3._gray(img_a), engine_v3._gray(img_b_resized), full=True)
inv = cv2.bitwise_not((diff * 255).clip(0, 255).astype(np.uint8))
_, thresh = cv2.threshold(inv, ssim_thresh, 255, cv2.THRESH_BINARY)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k)

bmask = np.zeros_like(thresh)
bmask[border_px:h - border_px, border_px:w - border_px] = 255

if valid_mask is not None:
    ep = engine_v3._erosion_px(h, w)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (ep, ep))
    bmask = cv2.bitwise_and(bmask, cv2.erode(valid_mask, k2))

# Apply OCR masking
engine_v3._mask_ocr_text(img_a, img_b_resized, bmask, line_mode=True)

thresh = cv2.bitwise_and(thresh, bmask)

cdiff = np.mean(cv2.absdiff(img_a, img_b_resized).astype(np.float32), axis=2)
delta_floor = engine_v3._dynamic_delta_floor(cdiff, valid_mask)
ga = engine_v3._gray(img_a).astype(np.float64)
gb = engine_v3._gray(img_b_resized).astype(np.float64)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours: {len(contours)}")

candidates = []
for idx, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < min_area:
        continue
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
    delta = cv2.mean(cdiff, mask=m)[0]
    
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    cx, cy, r = int(cx), int(cy), int(r)
    x1 = max(0, cx - r - 5); y1 = max(0, cy - r - 5)
    x2 = min(w, cx + r + 5); y2 = min(h, cy + r + 5)
    peak = float(np.abs(ga[y1:y2, x1:x2] - gb[y1:y2, x1:x2]).max())
    
    print(f"Contour {idx+1}: Center=({cx},{cy}), Area={area:.1f}, Delta={delta:.2f}, Peak={peak:.1f}")
    
    if delta < delta_floor:
        print("  -> Rejected: delta < delta_floor")
        continue
    if peak < peak_min:
        print("  -> Rejected: peak < peak_min")
        continue
        
    candidates.append((cx, cy, min(max(r + c_pad + 10, 20), max_r), delta))

print(f"\nCandidates after filtering: {len(candidates)}")
