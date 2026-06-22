import cv2
import numpy as np
import sys
from PIL import Image

sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

truth_actual = [
    (453.0, 210.4),   # 1. dam top-left rock
    (617.2, 503.4),   # 2. dam middle
    (211.5, 564.6),   # 3. water left
    (408.4, 791.5),   # 4. dam lower
    (953.1, 811.8),   # 5. dam right
    (387.2, 857.2),   # 6. dam lower-left
    (632.7, 870.7),   # 7. dam lower-middle
    (295.9, 899.5),   # 8. dam rock
    (809.1, 933.0),   # 9. dam lower-right
    (45.9, 457.7),    # 10. river patch left
    (44.7, 538.5),    # 11. rock left
    (1036.0, 430.0)   # 12. figure label at right edge
]

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

# Apply opening on difference maps before thresholding
k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
inv_opened = cv2.morphologyEx(inv, cv2.MORPH_OPEN, k_open)
lab_diff_opened = cv2.morphologyEx(lab_diff, cv2.MORPH_OPEN, k_open)

thresh_hue = (hue_diff_deg > std.HUE_FIXED_THRESH).astype(np.uint8) * 255
k_hue = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (std.HUE_DILATE_KSIZE, std.HUE_DILATE_KSIZE))
thresh_hue = cv2.morphologyEx(thresh_hue, cv2.MORPH_DILATE, k_hue)

valid_ssim = inv_opened[vy0:vy1, :]
valid_lab  = lab_diff_opened[vy0:vy1, :]
otsu_ssim  = cv2.threshold(valid_ssim, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
otsu_lab   = cv2.threshold(valid_lab,  0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]

_, thresh_ssim = cv2.threshold(inv_opened,      otsu_ssim, 255, cv2.THRESH_BINARY)
_, thresh_lab  = cv2.threshold(lab_diff_opened, otsu_lab,  255, cv2.THRESH_BINARY)

thresh = cv2.bitwise_or(thresh_ssim, thresh_lab)
thresh = cv2.bitwise_or(thresh,      thresh_hue)

bmask = np.ones((h, w), dtype=np.uint8) * 255
# Set tighter margin masks
bmask[:, :36] = 0
bmask[:, 1152-30:] = 0
bmask[:30, :] = 0
bmask[1006-30:, :] = 0

# Custom OCR masking (skips right edge label)
try:
    import pytesseract
    h_panel, w_panel = img_a.shape[:2]
    for img in (img_a, img_b_aligned):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        data = pytesseract.image_to_data(pil_img, config="-l eng+khm --psm 11", output_type=pytesseract.Output.DICT)
        n_boxes = len(data.get('level', []))
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if text:
                if not std.is_watermark_text(text):
                    continue
                x = data['left'][i]
                y = data['top'][i]
                w_box = data['width'][i]
                h_box = data['height'][i]
                if w_box > bmask.shape[1] * 0.35 or h_box > bmask.shape[0] * 0.25:
                    continue
                if x > w_panel - 180:
                    continue
                pad = 12
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(bmask.shape[1], x + w_box + pad)
                y2 = min(bmask.shape[0], y + h_box + pad)
                bmask[y1:y2, x1:x2] = 0
except Exception as e:
    pass

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
    (cx, cy), r = cv2.minEnclosingCircle(c)
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
    if cv2.contourArea(cnt) < 30:
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

candidates.sort(key=lambda x: -x[1])

# Sweep delta_floor
for df in [8.0, 10.0, 12.0, 15.0, 18.0]:
    surviving = [c for c in candidates if c[1] >= df]
    groups = []
    for cnt, delta, cx, cy, r in surviving:
        cx, cy = float(cx), float(cy)
        merged = False
        for grp in groups:
            if ((cx - grp[2]) ** 2 + (cy - grp[3]) ** 2) ** 0.5 < 55:
                grp[0].append((cx, cy, r))
                grp[1] = max(grp[1], delta)
                grp[2] = float(np.mean([s[0] for s in grp[0]]))
                grp[3] = float(np.mean([s[1] for s in grp[0]]))
                merged = True
                break
        if not merged:
            groups.append([[(cx, cy, r)], delta, cx, cy])
            
    if len(groups) >= 3:
        all_d = np.array([g[1] for g in groups])
        keep = []
        for grp in groups:
            others = all_d[all_d != grp[1]]
            med_oth = float(np.median(others)) if len(others) else grp[1]
            if grp[1] < std.LOW_DELTA_FRAC * med_oth:
                pass
            else:
                keep.append(grp)
        groups = keep

    matched_gt = set()
    fps = 0
    for grp in groups:
        cx, cy = grp[2], grp[3]
        closest_dist = float('inf')
        closest_gt_idx = -1
        for gt_idx, (tx, ty) in enumerate(truth_actual):
            dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_gt_idx = gt_idx
        if closest_dist < 60:
            matched_gt.add(closest_gt_idx)
        else:
            fps += 1
            
    print(f"df={df:.1f} -> Groups={len(groups)}, Matched={len(matched_gt)}/12, FP={fps}, Missed={[i+1 for i in range(12) if i not in matched_gt]}")
