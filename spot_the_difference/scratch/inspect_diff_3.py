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

std._mask_ocr_text(img_a, img_b_aligned, bmask)

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

print("Candidates BEFORE NMS:")
for i, c in enumerate(candidates):
    print(f"  Candidate {i+1}: center=({c[0]}, {c[1]}), r={c[2]}, delta={c[3]:.2f}")

candidates.sort(key=lambda x: -x[3])
kept = []
LINE_NMS_RADIUS = max(20, int(min(h, w) * 0.12))
for cx, cy, r, d in candidates:
    suppressed = False
    for kx, ky, _, kd in kept:
        dist = ((cx-kx)**2+(cy-ky)**2)**0.5
        if dist < LINE_NMS_RADIUS:
            suppressed = True
            print(f"  Candidate ({cx}, {cy}) suppressed by ({kx}, {ky}) [dist={dist:.1f} < {LINE_NMS_RADIUS}]")
            break
    if not suppressed:
        kept.append((cx, cy, r, d))
