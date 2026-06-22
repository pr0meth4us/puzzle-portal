import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

combined = std.load_bgr("/Users/nicksng/code/puzzle-portal/spot_the_difference/puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
img_b_aligned, valid_y_range, H_align = std.align(img_a, img_b, skip_ecc=False)

h, w = img_a.shape[:2]

# Let's run a sweep
gray_a = std._gray(img_a)
gray_b = std._gray(img_b_aligned)

score, diff = std.ssim(gray_a, gray_b, full=True)
inv = cv2.bitwise_not((diff * 255).clip(0, 255).astype(np.uint8))
lab_diff = std._lab_delta_map(img_a, img_b_aligned)
hue_diff_deg = std._hue_delta_map(img_a, img_b_aligned)

thresh_hue = (hue_diff_deg > std.HUE_FIXED_THRESH).astype(np.uint8) * 255
k_hue = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (std.HUE_DILATE_KSIZE, std.HUE_DILATE_KSIZE))
thresh_hue = cv2.morphologyEx(thresh_hue, cv2.MORPH_DILATE, k_hue)

vy0, vy1 = valid_y_range if valid_y_range else (0, h)
valid_ssim = inv[vy0:vy1, :]
valid_lab  = lab_diff[vy0:vy1, :]
otsu_ssim  = cv2.threshold(valid_ssim, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
otsu_lab   = cv2.threshold(valid_lab,  0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]

_, thresh_ssim = cv2.threshold(inv,      otsu_ssim, 255, cv2.THRESH_BINARY)
_, thresh_lab  = cv2.threshold(lab_diff, otsu_lab,  255, cv2.THRESH_BINARY)

thresh = cv2.bitwise_or(thresh_ssim, thresh_lab)
thresh = cv2.bitwise_or(thresh,      thresh_hue)

# Find margin spikes
gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
col_diff = np.mean(np.abs(gray[:, 1:].astype(float) - gray[:, :-1].astype(float)), axis=0)
row_diff = np.mean(np.abs(gray[1:, :].astype(float) - gray[:-1, :].astype(float)), axis=1)

max_search_x = min(180, max(30, int(w * 0.15)))
max_search_y = min(180, max(30, int(h * 0.15)))

left_spike = 0
for x in range(15, max_search_x):
    if col_diff[x] > 20.0:
        left_spike = x
        break
        
right_spike = w
for x in range(w - 16, w - max_search_x, -1):
    if col_diff[x] > 20.0:
        right_spike = x + 1
        break
        
top_spike = 0
for y in range(15, max_search_y):
    if row_diff[y] > 12.0:
        top_spike = y
        break
        
bottom_spike = h
for y in range(h - 16, h - max_search_y, -1):
    if row_diff[y] > 12.0:
        bottom_spike = y + 1
        break

# Sweep
for pad_val in [0, 2, 4]:
    bmask = np.ones((h, w), dtype=np.uint8) * 255
    if left_spike > 0:
        bmask[:, :left_spike + pad_val] = 0
    if right_spike < w:
        bmask[:, right_spike - pad_val:] = 0
    if top_spike > 0:
        bmask[:top_spike + pad_val, :] = 0
    if bottom_spike < h:
        bmask[bottom_spike - pad_val:, :] = 0
        
    if H_align is not None:
        mask_b = np.ones((h, w), dtype=np.uint8) * 255
        warped_mask_b = cv2.warpPerspective(mask_b, H_align, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        warped_mask_b = cv2.erode(warped_mask_b, kernel)
        bmask = cv2.bitwise_and(bmask, warped_mask_b)

    if vy0 > 16:
        bmask[:vy0, :] = 0
    if vy1 < h - 8:
        bmask[vy1:,  :] = 0

    std._mask_ocr_text(img_a, img_b_aligned, bmask)

    thresh_masked = cv2.bitwise_and(thresh, bmask)
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh_masked = cv2.morphologyEx(thresh_masked, cv2.MORPH_CLOSE, k9)
    thresh_masked = cv2.morphologyEx(thresh_masked, cv2.MORPH_OPEN,  k5)
    thresh_final = cv2.bitwise_and(thresh_masked, bmask)

    max_allowed_r = 70
    cdiff_rgb = np.mean(cv2.absdiff(img_a, img_b_aligned).astype(np.float32), axis=2)

    pre_cnts, _ = cv2.findContours(thresh_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    extra_candidates = []
    
    clean_thresh = thresh_final.copy()
    for c in pre_cnts:
        (_, _), r = cv2.minEnclosingCircle(c)
        if r > max_allowed_r:
            blob_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(blob_mask, [c], -1, 255, cv2.FILLED)
            subs = std._split_large_blob(blob_mask, cdiff_rgb, max_allowed_r, h, w)
            for scx, scy, sr, sd in subs:
                extra_candidates.append((scx, scy, sr, sd))
            cv2.drawContours(clean_thresh, [c], -1, 0, cv2.FILLED)

    contours, _ = cv2.findContours(clean_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
        rgb_delta = cv2.mean(cdiff_rgb, mask=m)[0]
        hue_delta = cv2.mean(hue_diff_deg, mask=m)[0]
        delta = max(rgb_delta, hue_delta * std.HUE_SCORE_WEIGHT)
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        candidates.append((cnt, delta, int(cx), int(cy), int(r)))

    for scx, scy, sr, sd in extra_candidates:
        candidates.append((None, sd, scx, scy, sr))

    candidates.sort(key=lambda x: -x[1])
    
    # Sweep delta_floor
    print(f"\n--- Pad={pad_val} ---")
    for df in [15.0, 18.0, 20.0, 22.0, 23.0, 24.0, 25.0]:
        surviving = [c for c in candidates if c[1] >= df]
        print(f"  delta_floor={df:.1f} -> {len(surviving)} differences")
