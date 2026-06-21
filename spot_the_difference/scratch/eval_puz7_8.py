import sys
import numpy as np
import cv2

sys.path.append("/Users/nicksng/code/random/spot_the_difference")
import spot_the_differences as std

class Candidate:
    def __init__(self, area, delta, cx, cy, r):
        self.area = area
        self.delta = delta
        self.cx = cx
        self.cy = cy
        self.r = r

def precompute_candidates(puz_path, name):
    combined = std.load_bgr(puz_path)
    cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
    img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
    is_vertical_split = (img_a.shape[0] == cropped_combined.shape[0])
    
    img_b_aligned, valid_y_range, H_align = std.align(img_a, img_b, skip_ecc=False)
    split_dir = "vertical" if is_vertical_split else "horizontal"
    
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

    hsv = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())
    is_colour = (mean_sat >= 20.0)

    border_val = max(16, int(min(h, w) * 0.02))
    bmask = np.zeros_like(thresh)
    
    if split_dir == "vertical":
        by_top, by_bot = 12, 20
        bmask[by_top:h - by_bot, 16:w - 10] = 255
    elif split_dir == "horizontal":
        by_top, by_bot = 16, 8
        bmask[by_top:h - by_bot, 8:w - 8] = 255
    else:
        by_top, by_bot = border_val, border_val
        bmask[by_top:h - by_bot, border_val:w - border_val] = 255
    
    if H_align is not None:
        mask_b = np.ones(img_b.shape[:2], dtype=np.uint8) * 255
        warped_mask_b = cv2.warpPerspective(mask_b, H_align, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        warped_mask_b = cv2.erode(warped_mask_b, kernel)
        bmask = cv2.bitwise_and(bmask, warped_mask_b)
        
    if vy0 > by_top:
        bmask[:vy0, :] = 0
    if vy1 < h - by_bot:
        bmask[vy1:,  :] = 0
    
    std._mask_margins(img_a, bmask)
    std._mask_ocr_text(img_a, img_b_aligned, bmask)

    # Scanned images -> edge_mask_ksize = 5
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    edges_a = cv2.Canny(gray_a, 50, 150)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_edges = cv2.dilate(edges_a, kernel_edge)
    bmask = cv2.bitwise_and(bmask, cv2.bitwise_not(dilated_edges))

    thresh = cv2.bitwise_and(thresh, bmask)
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k9)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k5)
    thresh = cv2.bitwise_and(thresh, bmask)

    max_allowed_r = int(min(h, w) * std.MAX_BLOB_RADIUS_FRAC)
    cdiff_rgb     = np.mean(cv2.absdiff(img_a, img_b_aligned).astype(np.float32), axis=2)

    pre_cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    extra_candidates = []
    
    # We will compute a new clean thresh for standard contours
    clean_thresh = thresh.copy()
    for c in pre_cnts:
        (_, _), r = cv2.minEnclosingCircle(c)
        if r > max_allowed_r:
            blob_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(blob_mask, [c], -1, 255, cv2.FILLED)
            subs = std._split_large_blob(blob_mask, cdiff_rgb, max_allowed_r, h, w)
            for scx, scy, sr, sd in subs:
                extra_candidates.append(Candidate(0, sd, scx, scy, sr)) # pseudo-contour area 0
            cv2.drawContours(clean_thresh, [c], -1, 0, cv2.FILLED)

    contours, _ = cv2.findContours(clean_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
        rgb_delta = cv2.mean(cdiff_rgb,    mask=m)[0]
        hue_delta = cv2.mean(hue_diff_deg, mask=m)[0]
        delta     = max(rgb_delta, hue_delta * std.HUE_SCORE_WEIGHT)
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        candidates.append(Candidate(area, delta, cx, cy, r))

    return candidates, extra_candidates, is_colour, max_allowed_r

def run_fast_sweep(candidates, extra_candidates, is_colour, max_allowed_r, target_count, name):
    print(f"\nSweeping parameters for {name}...")
    
    delta_floors = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 25.0, 30.0]
    min_areas = [20, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300]
    merge_radii = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80]
    
    matching_configs = []
    
    for ma in min_areas:
        # Filter standard candidates by area
        filtered_cnts = [c for c in candidates if c.area >= ma]
        all_candidates = filtered_cnts + extra_candidates
        
        if not all_candidates:
            continue
            
        deltas = sorted(c.delta for c in all_candidates)
        
        for df in delta_floors:
            floor = 10.0 if (df == 7.0 and is_colour) else df
            threshold, _ = std._auto_threshold(deltas, floor)
            surviving = [c for c in all_candidates if c.delta >= threshold]
            
            for mr in merge_radii:
                # Merge logic
                groups = []
                for c in surviving:
                    cx, cy, r, delta = float(c.cx), float(c.cy), c.r, c.delta
                    merged = False
                    for grp in groups:
                        if ((cx - grp[2]) ** 2 + (cy - grp[3]) ** 2) ** 0.5 < mr:
                            grp[0].append((cx, cy, r))
                            grp[1] = max(grp[1], delta)
                            grp[2] = float(np.mean([s[0] for s in grp[0]]))
                            grp[3] = float(np.mean([s[1] for s in grp[0]]))
                            merged = True
                            break
                    if not merged:
                        groups.append([[(cx, cy, r)], delta, cx, cy])
                
                # Drop low delta logic
                if len(groups) >= 3:
                    all_d = np.array([g[1] for g in groups])
                    keep = []
                    for grp in groups:
                        others = all_d[all_d != grp[1]]
                        med_oth = float(np.median(others)) if len(others) else grp[1]
                        if grp[1] >= std.LOW_DELTA_FRAC * med_oth:
                            keep.append(grp)
                    groups = keep
                
                if len(groups) == target_count:
                    matching_configs.append((df, ma, mr))
                    
    print(f"Found {len(matching_configs)} matching configurations.")
    for cfg in matching_configs[:20]:
        print(f"  delta_floor={cfg[0]}, min_area={cfg[1]}, merge_radius={cfg[2]}")

if __name__ == "__main__":
    c_7, ec_7, col_7, max_r_7 = precompute_candidates("/Users/nicksng/code/random/spot_the_difference/puzzles/puzzle_07.jpg", "Puzzle 7")
    run_fast_sweep(c_7, ec_7, col_7, max_r_7, 12, "Puzzle 7")

    c_8, ec_8, col_8, max_r_8 = precompute_candidates("/Users/nicksng/code/random/spot_the_difference/puzzles/puzzle_08.jpg", "Puzzle 8")
    run_fast_sweep(c_8, ec_8, col_8, max_r_8, 10, "Puzzle 8")
