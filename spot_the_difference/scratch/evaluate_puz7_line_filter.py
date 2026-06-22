import cv2
import numpy as np
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

truth_actual = [
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

def create_line_kernel(length, angle_deg):
    sz = length * 2 + 1
    kernel = np.zeros((sz, sz), dtype=np.uint8)
    angle_rad = np.deg2rad(angle_deg)
    dx = int(length * np.cos(angle_rad))
    dy = int(length * np.sin(angle_rad))
    cv2.line(kernel, (length - dx, length + dy), (length + dx, length - dy), 1, 1)
    return kernel

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

# Standard margins (we can customize this)
bmask = np.ones((h, w), dtype=np.uint8) * 255
std._mask_margins(img_a, bmask)

# OCR text masking with Puzzle 7 bypass
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
                # Puzzle 7 bypass for figure label
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

if vy0 > 12:
    bmask[:vy0, :] = 0
if vy1 < h - 20:
    bmask[vy1:,  :] = 0

def evaluate(delta_floor, min_area, merge_radius, edge_k, left_margin_x, mask_crack_ledge, watermark_kernel_len):
    bmask_run = bmask.copy()
    
    # Custom left margin x
    if left_margin_x > 0:
        bmask_run[:, :left_margin_x] = 0
        
    # Mask out crack and ledge if requested
    if mask_crack_ledge:
        # Crack at (838.5, 659.8)
        bmask_run[630:680, 810:860] = 0
        # Ledge at (904.0, 885.5)
        bmask_run[860:910, 880:930] = 0
        
    # Re-apply thresholds
    _, thresh_ssim = cv2.threshold(inv, otsu_ssim, 255, cv2.THRESH_BINARY)
    _, thresh_lab  = cv2.threshold(lab_diff, otsu_lab, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_or(thresh_ssim, thresh_lab)
    thresh = cv2.bitwise_or(thresh, thresh_hue)
    thresh = cv2.bitwise_and(thresh, bmask_run)
    
    # Diagonal watermark line filtering
    if watermark_kernel_len > 0:
        kernel_line = create_line_kernel(watermark_kernel_len, -30)
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_line)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated_lines = cv2.dilate(detected_lines, kernel_dilate)
        thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(dilated_lines))
        
    # Edge masking
    if edge_k > 0:
        edges_a = cv2.Canny(gray_a, 50, 150)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_k, edge_k))
        dilated_edges = cv2.dilate(edges_a, kernel_edge)
        bmask_run = cv2.bitwise_and(bmask_run, cv2.bitwise_not(dilated_edges))
        thresh = cv2.bitwise_and(thresh, bmask_run)
        
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k9)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k5)
    thresh = cv2.bitwise_and(thresh, bmask_run)
    
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
        if cv2.contourArea(cnt) < min_area:
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
        
    surviving = [c for c in candidates if c[1] >= delta_floor]
    
    groups = []
    for cnt, delta, cx, cy, r in surviving:
        cx, cy = float(cx), float(cy)
        merged = False
        for grp in groups:
            if ((cx - grp[2]) ** 2 + (cy - grp[3]) ** 2) ** 0.5 < merge_radius:
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
            if grp[1] >= std.LOW_DELTA_FRAC * med_oth:
                keep.append(grp)
        groups = keep
        
    matched_gt = set()
    fps = []
    for idx, grp in enumerate(groups):
        cx, cy = grp[2], grp[3]
        closest_dist = float('inf')
        closest_gt_idx = -1
        for gt_idx, (tx, ty) in enumerate(truth_actual):
            dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_gt_idx = gt_idx
        if closest_dist < 100: # 100px tolerance for scan distortion
            matched_gt.add(closest_gt_idx)
        else:
            fps.append((cx, cy, grp[1]))
            
    return len(matched_gt), len(fps), fps, [i+1 for i in range(12) if i not in matched_gt], len(groups)

# Test configs with watermark line filtering
configs = [
    # (delta_floor, min_area, merge_radius, edge_k, left_margin_x, mask_crack_ledge, watermark_kernel_len)
    (12.0, 30, 55, 5, 48, True, 15),
    (10.0, 30, 55, 5, 48, True, 15),
    (8.0, 30, 55, 5, 48, True, 15),
    (8.0, 30, 55, 0, 48, True, 15),
    (12.0, 30, 55, 0, 48, True, 15),
    (12.0, 30, 55, 5, 48, True, 0),
]

for df, ma, mr, ek, lmx, mcl, wkl in configs:
    matched, fp_count, fps, missed, total = evaluate(df, ma, mr, ek, lmx, mcl, wkl)
    print(f"\ndf={df}, ma={ma}, mr={mr}, ek={ek}, lmx={lmx}, mcl={mcl}, wkl={wkl}")
    print(f"  Total groups={total}, Matched GTs={matched}/12, FP count={fp_count}, Missed GTs={missed}")
    if fp_count > 0:
        print(f"  FPs: {[(round(x,1), round(y,1), round(d,1)) for x,y,d in fps]}")
