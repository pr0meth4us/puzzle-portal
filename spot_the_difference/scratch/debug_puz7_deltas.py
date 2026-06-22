import cv2
import numpy as np
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

combined = std.load_bgr("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
img_b_aligned, valid_y_range, H_align = std.align(img_a, img_b, skip_ecc=False)

h, w = img_a.shape[:2]
vy0, vy1 = valid_y_range if valid_y_range else (0, h)
vy0, vy1 = max(0, vy0), min(h, vy1)

gray_a = std._gray(img_a)
gray_b = std._gray(img_b_aligned)

score, diff = std.ssim(gray_a, gray_b, full=True)
inv = cv2.bitwise_not((diff * 255).clip(0, 255).astype(np.uint8))
lab_diff = std._lab_delta_map(img_a, img_b_aligned)
hue_diff_deg = std._hue_delta_map(img_a, img_b_aligned)

thresh_hue = (hue_diff_deg > std.HUE_FIXED_THRESH).astype(np.uint8) * 255
k_hue = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (std.HUE_DILATE_KSIZE, std.HUE_DILATE_KSIZE))
thresh_hue = cv2.morphologyEx(thresh_hue, cv2.MORPH_DILATE, k_hue)

valid_ssim = inv[vy0:vy1, :]
valid_lab  = lab_diff[vy0:vy1, :]
otsu_ssim  = cv2.threshold(valid_ssim, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
otsu_lab   = cv2.threshold(valid_lab,  0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]

_, thresh_ssim = cv2.threshold(inv,      otsu_ssim, 255, cv2.THRESH_BINARY)
_, thresh_lab  = cv2.threshold(lab_diff, otsu_lab,  255, cv2.THRESH_BINARY)

thresh = cv2.bitwise_or(thresh_ssim, thresh_lab)
thresh = cv2.bitwise_or(thresh,      thresh_hue)

bmask = np.ones((h, w), dtype=np.uint8) * 255

# Apply margins
std._mask_margins(img_a, bmask)
std._mask_ocr_text(img_a, img_b_aligned, bmask)

thresh = cv2.bitwise_and(thresh, bmask)
k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k9)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k5)
thresh = cv2.bitwise_and(thresh, bmask)

# Find candidates
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cdiff_rgb = np.mean(cv2.absdiff(img_a, img_b_aligned).astype(np.float32), axis=2)

print("\nCandidate Contours and their Deltas:")
candidates = []
for idx, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < 30:
        continue
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
    rgb_delta = cv2.mean(cdiff_rgb, mask=m)[0]
    hue_delta = cv2.mean(hue_diff_deg, mask=m)[0]
    delta = max(rgb_delta, hue_delta * std.HUE_SCORE_WEIGHT)
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    candidates.append((idx, int(cx), int(cy), int(r), area, delta))

candidates.sort(key=lambda x: -x[5])
for idx, cx, cy, r, area, delta in candidates:
    print(f"  Cand {idx}: Center=({cx}, {cy}), r={r}, Area={area:.1f}, Delta={delta:.2f}")
