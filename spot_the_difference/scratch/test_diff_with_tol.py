import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from skimage.metrics import structural_similarity as ssim
from spot_the_differences import load_bgr, crop_text_by_gap, auto_slice, align, _gray, _hue_delta_map, _split_large_blob, _auto_threshold, _mask_margins, _mask_ocr_text, random_run_color, build_stacked_output, add_watermark

def detect_with_tolerance(img_a: np.ndarray,
                          img_b: np.ndarray,
                          min_area: int      = 50,
                          delta_floor: float = 7.0,
                          valid_y_range=None,
                          split_dir=None,
                          H=None,
                          ksize=5):
    h, w = img_a.shape[:2]
    vy0, vy1 = valid_y_range if valid_y_range else (0, h)
    vy0, vy1 = max(0, vy0), min(h, vy1)

    gray_a = _gray(img_a)
    gray_b = _gray(img_b)

    # 1. Compute SSIM inv
    score, diff = ssim(gray_a, gray_b, full=True)
    inv = cv2.bitwise_not((diff * 255).clip(0, 255).astype(np.uint8))

    # 2. Compute Lab diff
    # We will compute morphological tolerance filtered Lab diff
    la = cv2.cvtColor(img_a, cv2.COLOR_BGR2LAB)
    lb = cv2.cvtColor(img_b, cv2.COLOR_BGR2LAB)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    
    # Filter for grayscale / inv
    gray_a_min = cv2.erode(gray_a, kernel)
    gray_a_max = cv2.dilate(gray_a, kernel)
    gray_b_min = cv2.erode(gray_b, kernel)
    gray_b_max = cv2.dilate(gray_b, kernel)
    diff_b_to_a = np.maximum(0, gray_b.astype(float) - gray_a_max.astype(float)) + np.maximum(0, gray_a_min.astype(float) - gray_b.astype(float))
    diff_a_to_b = np.maximum(0, gray_a.astype(float) - gray_b_max.astype(float)) + np.maximum(0, gray_b_min.astype(float) - gray_a.astype(float))
    tol_gray_diff = np.minimum(diff_b_to_a, diff_a_to_b).astype(np.uint8)
    
    # Mask inv using the tolerance diff mask
    _, tol_gray_mask = cv2.threshold(tol_gray_diff, 5, 255, cv2.THRESH_BINARY)
    inv = cv2.bitwise_and(inv, tol_gray_mask)

    # Filter for Lab
    lab_diff_channels = []
    for c in range(3):
        a_chan = la[:, :, c]
        b_chan = lb[:, :, c]
        a_min = cv2.erode(a_chan, kernel)
        a_max = cv2.dilate(a_chan, kernel)
        b_min = cv2.erode(b_chan, kernel)
        b_max = cv2.dilate(b_chan, kernel)
        
        c_diff_b_to_a = np.maximum(0, b_chan.astype(float) - a_max.astype(float)) + np.maximum(0, a_min.astype(float) - b_chan.astype(float))
        c_diff_a_to_b = np.maximum(0, a_chan.astype(float) - b_max.astype(float)) + np.maximum(0, b_min.astype(float) - a_chan.astype(float))
        lab_diff_channels.append(np.minimum(c_diff_b_to_a, c_diff_a_to_b))
        
    combined_lab_tol = np.sqrt(sum(d**2 for d in lab_diff_channels))
    mx = combined_lab_tol.max()
    lab_diff = (combined_lab_tol / mx * 255).astype(np.uint8) if mx > 0 else combined_lab_tol.astype(np.uint8)

    # 3. Hue diff
    hue_diff_deg = _hue_delta_map(img_a, img_b)
    # Filter hue diff by the same grayscale tolerance mask (or we can do circular hue tolerance, but gray tolerance mask is extremely robust)
    thresh_hue   = (hue_diff_deg > 30).astype(np.uint8) * 255
    k_hue = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh_hue = cv2.morphologyEx(thresh_hue, cv2.MORPH_DILATE, k_hue)
    thresh_hue = cv2.bitwise_and(thresh_hue, tol_gray_mask)

    # Thresholding using Otsu on the filtered outputs
    valid_ssim = inv[vy0:vy1, :]
    valid_lab  = lab_diff[vy0:vy1, :]
    otsu_ssim  = cv2.threshold(valid_ssim, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
    otsu_lab   = cv2.threshold(valid_lab,  0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
    _, thresh_ssim = cv2.threshold(inv,      otsu_ssim, 255, cv2.THRESH_BINARY)
    _, thresh_lab  = cv2.threshold(lab_diff, otsu_lab,  255, cv2.THRESH_BINARY)

    print(f"[INFO] Otsu SSIM={otsu_ssim:.0f}  Lab={otsu_lab:.0f}")

    thresh = cv2.bitwise_or(thresh_ssim, thresh_lab)
    thresh = cv2.bitwise_or(thresh,      thresh_hue)

    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k9)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k5)

    hsv = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())
    is_colour = (mean_sat >= 20.0)
    floor = 10.0 if (delta_floor == 7.0 and is_colour) else delta_floor

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
    
    if H is not None:
        mask_b = np.ones(img_b.shape[:2], dtype=np.uint8) * 255
        warped_mask_b = cv2.warpPerspective(mask_b, H, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        warped_mask_b = cv2.erode(warped_mask_b, kernel_e)
        bmask = cv2.bitwise_and(bmask, warped_mask_b)
        
    if vy0 > by_top:
        bmask[:vy0, :] = 0
    if vy1 < h - by_bot:
        bmask[vy1:,  :] = 0
    
    _mask_margins(img_a, bmask)
    _mask_ocr_text(img_a, img_b, bmask)
    
    # Apply bmask before AND after morphology to prevent border noise from bleeding
    thresh = cv2.bitwise_and(thresh, bmask)

    max_allowed_r = int(min(h, w) * 0.25) # MAX_BLOB_RADIUS_FRAC = 0.25
    cdiff_rgb     = np.mean(cv2.absdiff(img_a, img_b).astype(np.float32), axis=2)

    # Split oversized blobs
    pre_cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    extra_candidates = []
    for c in pre_cnts:
        (_, _), r = cv2.minEnclosingCircle(c)
        if r > max_allowed_r:
            blob_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(blob_mask, [c], -1, 255, cv2.FILLED)
            subs = _split_large_blob(blob_mask, cdiff_rgb, max_allowed_r, h, w)
            for scx, scy, sr, sd in subs:
                extra_candidates.append((scx, scy, sr, sd))
            cv2.drawContours(thresh, [c], -1, 0, cv2.FILLED)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
        rgb_delta = cv2.mean(cdiff_rgb,    mask=m)[0]
        hue_delta = cv2.mean(hue_diff_deg, mask=m)[0]
        delta     = max(rgb_delta, hue_delta * 0.5)
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        candidates.append((cnt, delta, int(cx), int(cy), int(r)))

    for scx, scy, sr, sd in extra_candidates:
        candidates.append((None, sd, scx, scy, sr))

    if not candidates:
        return [], 0

    deltas = sorted(d for _, d, _, _, _ in candidates)
    threshold, reason = _auto_threshold(deltas, floor)
    print(f"[INFO] Candidates : {len(candidates)}  deltas: {[round(d, 1) for d in deltas]}")
    print(f"[INFO] Delta-threshold : {threshold:.1f}  ({reason})")

    surviving = [(cnt, delta, cx, cy, r)
                 for cnt, delta, cx, cy, r in candidates
                 if delta >= threshold]

    groups = []
    for cnt, delta, cx, cy, r in surviving:
        cx, cy = float(cx), float(cy)
        merged = False
        for grp in groups:
            if ((cx - grp[2]) ** 2 + (cy - grp[3]) ** 2) ** 0.5 < 55: # MERGE_RADIUS = 55
                grp[0].append((cx, cy, r))
                grp[1] = max(grp[1], delta)
                grp[2] = float(np.mean([s[0] for s in grp[0]]))
                grp[3] = float(np.mean([s[1] for s in grp[0]]))
                merged = True
                break
        if not merged:
            groups.append([[(cx, cy, r)], delta, cx, cy])

    print(f"[INFO] After merging: {len(groups)} groups")
    print("Group deltas:", [round(g[1], 1) for g in groups])
    groups.sort(key=lambda g: g[1], reverse=True)

    if len(groups) >= 3:
        all_d = np.array([g[1] for g in groups])
        keep  = []
        for grp in groups:
            others  = all_d[all_d != grp[1]]
            med_oth = float(np.median(others)) if len(others) else grp[1]
            if grp[1] < 0.3 * med_oth:
                print(f"[INFO] Dropping low-delta group delta={grp[1]:.1f}")
            else:
                keep.append(grp)
        groups = keep

    circles = []
    for grp in groups:
        sub       = grp[0]
        centres   = np.array([[s[0], s[1]] for s in sub], dtype=np.float32)
        max_sub_r = max(s[2] for s in sub)
        if len(centres) == 1:
            cx, cy = centres[0]
            r = max(int(max_sub_r) + 6, 18)
        else:
            (cx, cy), span = cv2.minEnclosingCircle(centres.reshape(-1, 1, 2))
            r = max(int(span + max_sub_r) + 6, 18)
        if r > max_allowed_r:
            r = max_allowed_r
        circles.append((int(cx), int(cy), r))

    return circles, len(circles)

def main():
    for p_num in ("07", "08"):
        print(f"\n--- TESTING PUZZLE {p_num} WITH TOLERANCE DIFF ---")
        combined = load_bgr(f"puzzles/puzzle_{p_num}.jpg")
        cropped_combined, crop_y_offset = crop_text_by_gap(combined)
        img_a, img_b, a_start, b_start = auto_slice(cropped_combined)
        img_b_aligned, valid_y_range, H_align = align(img_a, img_b, skip_ecc=False)
        split_dir = "horizontal"
        
        circles, count = detect_with_tolerance(
            img_a, img_b_aligned,
            min_area=50,
            delta_floor=7.0,
            valid_y_range=valid_y_range,
            split_dir=split_dir,
            H=H_align,
            ksize=3
        )
        print(f"Differences found: {count}")
        for idx, (cx, cy, r) in enumerate(circles):
            print(f"  Circle {idx}: ({cx}, {cy}), r={r}")

if __name__ == "__main__":
    main()
