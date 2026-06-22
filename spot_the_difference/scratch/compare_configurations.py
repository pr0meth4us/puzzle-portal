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

# Let's load the ground truth circles from answer_07
ans_img = cv2.imread("/Users/nicksng/code/puzzle-portal/spot_the_difference/correct_answers/answer_07.jpg")
h_ans, w_ans = ans_img.shape[:2]
scale_x = w / w_ans
scale_y = (h * 2 + 36) / h_ans # wait, original height is 2048, h_ans is 853

# Let's get the 12 top-panel circles from find_red_circles_puz7.py
truth_in_a = [
    (808.1, 931.6),
    (294.2, 900.0),
    (633.1, 868.3),
    (387.6, 856.3),
    (952.8, 811.2),
    (408.0, 789.6),
    (212.4, 562.8),
    (616.8, 502.8),
    (45.6, 457.2), # River patch!
    (169.2, 351.6),
    (132.0, 218.2), # Rock!
    (452.9, 210.7)
]

# Adjust y coordinates to img_a (subtract crop_y_offset = 36)
truth_adjusted = [(tx, ty - 36) for tx, ty in truth_in_a]

def run_config(pad_val, df_val):
    print(f"\nEvaluating configuration: pad={pad_val}, delta_floor={df_val}")
    
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

    bmask = np.ones((h, w), dtype=np.uint8) * 255

    # Margins on A
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
    surviving = [c for c in candidates if c[1] >= df_val]
    
    # Check matches with Ground Truth
    matched_gt = set()
    matches_info = []
    
    for idx, (_, delta, cx, cy, r) in enumerate(surviving):
        # Find closest ground truth
        closest_dist = float('inf')
        closest_gt_idx = -1
        for g_idx, (tx, ty) in enumerate(truth_adjusted):
            dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_gt_idx = g_idx
        
        matches_info.append((idx, cx, cy, r, delta, closest_gt_idx, closest_dist))
        if closest_dist < 60: # Match threshold
            matched_gt.add(closest_gt_idx)
            
    print(f"Matched {len(matched_gt)} / 12 Ground Truth circles.")
    print("Matches details:")
    print("Cand | Center     | r  | delta | GT idx | dist to GT | Status")
    print("-" * 70)
    for idx, cx, cy, r, delta, gt_idx, dist in matches_info:
        status = f"MATCH (GT {gt_idx})" if dist < 60 else "FALSE POSITIVE"
        print(f"{idx:4d} | ({cx:4d}, {cy:4d}) | {r:2d} | {delta:5.2f} | {gt_idx:6d} | {dist:10.1f} | {status}")
        
    missed = [g_idx for g_idx in range(12) if g_idx not in matched_gt]
    print(f"Missed GT indices: {missed}")
    for m_idx in missed:
        print(f"  Missed GT {m_idx}: {truth_adjusted[m_idx]}")

run_config(2, 24.0)
run_config(4, 25.0)
