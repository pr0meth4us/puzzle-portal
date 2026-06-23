import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import spot_the_difference.engine_v3 as eng

p_path1 = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference/puzzles/puzzle_extra_05.jpg")
p_path2 = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference/puzzles/puzzle_extra_06.jpg")
img_a = cv2.imread(str(p_path1))
img_b = cv2.imread(str(p_path2))

img_b_aligned, H_align, valid_mask = eng.align(img_a, img_b, skip_ecc=True)

h, w = img_a.shape[:2]
min_area   = eng._min_area(w, h)
max_r      = int(min(h, w) * 0.15)
c_pad      = eng._circle_pad(w)
border_px  = max(8, int(min(h, w) * 0.02))

# Mapped ground truth
gt_coords = [
    (120, 64),
    (556, 85),
    (304, 88),
    (449, 100),
    (122, 128),
    (218, 219),
    (568, 292),
    (66, 314),
    (332, 365),
    (528, 396)
]

# Run SSIM once
score, diff = eng.ssim(eng._gray(img_a), eng._gray(img_b_aligned), full=True)
inv = cv2.bitwise_not((diff * 255).clip(0, 255).astype(np.uint8))

# Build raw bmask once
bmask_base = np.zeros(inv.shape, dtype=np.uint8)
bmask_base[border_px:h - border_px, border_px:w - border_px] = 255
if valid_mask is not None:
    ep = eng._erosion_px(h, w)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (ep, ep))
    bmask_base = cv2.bitwise_and(bmask_base, cv2.erode(valid_mask, k2))

# OCR mask once
eng._mask_ocr_text(img_a, img_b_aligned, bmask_base, line_mode=True)

cdiff = np.mean(cv2.absdiff(img_a, img_b_aligned).astype(np.float32), axis=2)
delta_floor = eng._dynamic_delta_floor(cdiff, valid_mask)
ga = eng._gray(img_a).astype(np.float64)
gb = eng._gray(img_b_aligned).astype(np.float64)

# Let's search over a wider parameter space
for ssim_thresh in [30, 40, 50, 60]:
    _, thresh_base = cv2.threshold(inv, ssim_thresh, 255, cv2.THRESH_BINARY)
    for morph_k in [3, 5]:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        thresh = cv2.morphologyEx(thresh_base, cv2.MORPH_CLOSE, k)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k)
        thresh = cv2.bitwise_and(thresh, bmask_base)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates_raw = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
            delta = cv2.mean(cdiff, mask=m)[0]
            if delta < delta_floor:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            cx, cy, r = int(cx), int(cy), int(r)
            x1 = max(0, cx - r - 5); y1 = max(0, cy - r - 5)
            x2 = min(w, cx + r + 5); y2 = min(h, cy + r + 5)
            peak = float(np.abs(ga[y1:y2, x1:x2] - gb[y1:y2, x1:x2]).max())
            candidates_raw.append((cx, cy, r, peak))

        for peak_min in [100, 120, 140, 160, 180, 200, 220]:
            candidates = [c for c in candidates_raw if c[3] >= peak_min]
            candidates.sort(key=lambda x: -x[3])

            for nms_r in range(20, 60, 4):
                kept = []
                for cx, cy, r, p in candidates:
                    if not any(((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5 < nms_r for kx, ky, _, _ in kept):
                        kept.append((cx, cy, r, p))
                        
                matches = 0
                for gtx, gty in gt_coords:
                    if any(((cx - gtx) ** 2 + (cy - gty) ** 2) ** 0.5 < 50 for cx, cy, _, _ in kept):
                        matches += 1
                
                # Check for Kept <= 12 and Matches >= 7
                if len(kept) <= 12 and matches >= 7:
                    print(f"ssim_thresh={ssim_thresh} morph_k={morph_k} peak_min={peak_min} nms_r={nms_r} -> Kept: {len(kept)} | Matches: {matches}/{len(gt_coords)}")
