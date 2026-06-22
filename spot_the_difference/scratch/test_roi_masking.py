import cv2
import numpy as np
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

truth_refined = [
    (456.9, 201.1),   # 1. dam top-left rock
    (602.3, 512.4),   # 2. dam middle
    (216.4, 572.8),   # 3. water left
    (404.4, 809.4),   # 4. dam lower
    (948.1, 814.5),   # 5. dam right
    (391.3, 819.0),   # 6. dam lower-left
    (636.2, 864.9),   # 7. dam lower-middle
    (270.9, 923.6),   # 8. dam rock
    (827.2, 907.0),   # 9. dam lower-right
    (72.5, 442.0),    # 10. river patch left
    (71.8, 534.6),    # 11. rock left
    (1060.9, 448.4)   # 12. figure label at right edge
]

puz = cv2.imread("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(puz)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
img_b_aligned, valid_y_range, H_align = std.align(img_a, img_b, skip_ecc=False)

h, w = img_a.shape[:2]
vy0, vy1 = valid_y_range if valid_y_range else (0, h)

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

# Standard thresholding
_, thresh_ssim = cv2.threshold(inv, otsu_ssim, 255, cv2.THRESH_BINARY)
_, thresh_lab  = cv2.threshold(lab_diff, otsu_lab, 255, cv2.THRESH_BINARY)
thresh = cv2.bitwise_or(thresh_ssim, thresh_lab)
thresh = cv2.bitwise_or(thresh, thresh_hue)

# Construct ROI-only bmask
bmask = np.zeros((h, w), dtype=np.uint8)
for cx, cy in truth_refined:
    cv2.circle(bmask, (int(cx), int(cy)), 45, 255, -1)

thresh = cv2.bitwise_and(thresh, bmask)

k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k9)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k5)
thresh = cv2.bitwise_and(thresh, bmask)

max_allowed_r = int(min(h, w) * std.MAX_BLOB_RADIUS_FRAC)
cdiff_rgb = np.mean(cv2.absdiff(img_a, img_b_aligned).astype(np.float32), axis=2)

pre_cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
extra_candidates = []
clean_thresh = thresh.copy()
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
    if cv2.contourArea(cnt) < 15: # slightly lower min_area to capture small parts
        continue
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
    rgb_delta = cv2.mean(cdiff_rgb, mask=m)[0]
    hue_delta = cv2.mean(hue_diff_deg, mask=m)[0]
    delta = max(rgb_delta, hue_delta * std.HUE_SCORE_WEIGHT)
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    candidates.append((cnt, delta, int(cx), int(cy), int(r)))
    
for scx, scy, sr, sd in extra_candidates:
    candidates.append((None, sd, scx, scy, sr))
    
# Lower floor to 5.0 since we only look at the 12 difference ROIs anyway
surviving = [c for c in candidates if c[1] >= 5.0]

# Merge with small radius (14) to avoid merging GT 4 and 6
groups = []
for cnt, delta, cx, cy, r in surviving:
    cx, cy = float(cx), float(cy)
    merged = False
    for grp in groups:
        if ((cx - grp[2]) ** 2 + (cy - grp[3]) ** 2) ** 0.5 < 14:
            grp[0].append((cx, cy, r))
            grp[1] = max(grp[1], delta)
            grp[2] = float(np.mean([s[0] for s in grp[0]]))
            grp[3] = float(np.mean([s[1] for s in grp[0]]))
            merged = True
            break
    if not merged:
        groups.append([[(cx, cy, r)], delta, cx, cy])

print("ROI Masked Results:")
print("Total groups found:", len(groups))
matched_gt = set()
for idx, grp in enumerate(groups):
    cx, cy = grp[2], grp[3]
    closest_dist = float('inf')
    closest_gt_idx = -1
    for gt_idx, (tx, ty) in enumerate(truth_refined):
        dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
        if dist < closest_dist:
            closest_dist = dist
            closest_gt_idx = gt_idx
    if closest_dist < 60:
        matched_gt.add(closest_gt_idx)
        print(f"  Group {idx+1}: Center=({cx:.1f}, {cy:.1f}), Delta={grp[1]:.2f} -> MATCHED GT {closest_gt_idx+1}")
    else:
        print(f"  Group {idx+1}: Center=({cx:.1f}, {cy:.1f}), Delta={grp[1]:.2f} -> FALSE POSITIVE (dist={closest_dist:.1f})")

print(f"\nMatched {len(matched_gt)}/12 GTs. Missed: {[i+1 for i in range(12) if i not in matched_gt]}")
