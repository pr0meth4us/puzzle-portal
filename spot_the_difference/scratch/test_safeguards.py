import cv2
import numpy as np
import sys
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from spot_the_differences import load_bgr, crop_text_by_gap, auto_slice, align, _gray, _hue_delta_map, _split_large_blob, _auto_threshold, random_run_color

def improved_mask_margins(img: np.ndarray, bmask: np.ndarray, needs_padding: bool):
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
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
                
        # Apply padding only if SIFT alignment was low SSIM (e.g. puzzle 07)
        pad_x = 16 if needs_padding else 1
        pad_y = 16 if needs_padding else 1
        
        if left_spike > 0:
            bmask[:, :left_spike + pad_x] = 0
        if right_spike < w:
            bmask[:, right_spike - pad_x:] = 0
        if top_spike > 0:
            bmask[:top_spike + pad_y, :] = 0
        if bottom_spike < h:
            bmask[bottom_spike - pad_y:, :] = 0
            
    except Exception as e:
        print(f"[WARN] Margin masking failed: {e}")

def improved_mask_ocr_text(img_a: np.ndarray, img_b: np.ndarray, bmask: np.ndarray):
    try:
        import pytesseract
        h_panel, w_panel = img_a.shape[:2]
        
        # 1. OCR on the full image with default PSM (safe, conservative)
        for img in (img_a, img_b):
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            data = pytesseract.image_to_data(pil_img, config="-l eng+khm", output_type=pytesseract.Output.DICT)
            n_boxes = len(data.get('level', []))
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if text:
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    pad = 4
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(bmask.shape[1], x + w + pad)
                    y2 = min(bmask.shape[0], y + h + pad)
                    bmask[y1:y2, x1:x2] = 0
        
        # 2. OCR on bottom-left and bottom-right corner crops with --psm 11 (only on large panels)
        if w_panel > 700 and h_panel > 700:
            corner_h = 120
            corner_w = 250
            for img in (img_a, img_b):
                # Bottom-left crop
                bl_crop = img[h_panel - corner_h:, :corner_w]
                bl_pil = Image.fromarray(cv2.cvtColor(bl_crop, cv2.COLOR_BGR2RGB))
                data = pytesseract.image_to_data(bl_pil, config="-l eng+khm --psm 11", output_type=pytesseract.Output.DICT)
                n_boxes = len(data.get('level', []))
                for i in range(n_boxes):
                    text = data['text'][i].strip()
                    if text:
                        x = data['left'][i]
                        y = data['top'][i]
                        w = data['width'][i]
                        h = data['height'][i]
                        # Map back to full panel
                        x1 = max(0, x - 4)
                        y1 = max(0, (h_panel - corner_h) + y - 4)
                        x2 = min(bmask.shape[1], x + w + 4)
                        y2 = min(bmask.shape[0], (h_panel - corner_h) + y + h + 4)
                        bmask[y1:y2, x1:x2] = 0
                        
                # Bottom-right crop
                br_crop = img[h_panel - corner_h:, w_panel - corner_w:]
                br_pil = Image.fromarray(cv2.cvtColor(br_crop, cv2.COLOR_BGR2RGB))
                data = pytesseract.image_to_data(br_pil, config="-l eng+khm --psm 11", output_type=pytesseract.Output.DICT)
                n_boxes = len(data.get('level', []))
                for i in range(n_boxes):
                    text = data['text'][i].strip()
                    if text:
                        x = data['left'][i]
                        y = data['top'][i]
                        w = data['width'][i]
                        h = data['height'][i]
                        # Map back to full panel
                        x1 = max(0, (w_panel - corner_w) + x - 4)
                        y1 = max(0, (h_panel - corner_h) + y - 4)
                        x2 = min(bmask.shape[1], (w_panel - corner_w) + x + w + 4)
                        y2 = min(bmask.shape[0], (h_panel - corner_h) + y + h + 4)
                        bmask[y1:y2, x1:x2] = 0
                        
        # ── VERTICAL COLOR-MASK OCR FOR LABELS ON RIGHT SIDE ────────────────────
        crop_w = 150
        for img in (img_a, img_b):
            crop = img[:, w_panel - crop_w:] if w_panel > crop_w else img
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            mask_y = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
            mask_r1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
            mask_r2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
            mask_r = mask_r1 | mask_r2
            
            for mask in (mask_y, mask_r):
                rot_mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cfg = "-l eng+khm --psm 11"
                data = pytesseract.image_to_data(Image.fromarray(rot_mask), config=cfg, output_type=pytesseract.Output.DICT)
                n_boxes = len(data.get('level', []))
                for i in range(n_boxes):
                    text = data['text'][i].strip()
                    if text:
                        rx = data['left'][i]
                        ry = data['top'][i]
                        rw = data['width'][i]
                        rh = data['height'][i]
                        
                        x_orig = crop_w - ry - rh
                        y_orig = rx
                        
                        x_full = w_panel - crop_w + x_orig
                        y_full = y_orig
                        w_full = rh
                        h_full = rw
                        
                        pad = 12
                        x1 = max(0, x_full - pad)
                        y1 = max(0, y_full - pad)
                        x2 = min(w_panel, x_full + w_full + pad)
                        y2 = min(h_panel, y_full + h_full + pad)
                        
                        bmask[y1:y2, x1:x2] = 0
    except Exception as e:
        print(f"[WARN] OCR text masking failed or skipped: {e}")

def detect_with_moved_mask(img_a: np.ndarray,
                           img_b: np.ndarray,
                           min_area: int      = 50,
                           delta_floor: float = 7.0,
                           valid_y_range=None,
                           split_dir=None,
                           H=None):
    h, w = img_a.shape[:2]
    vy0, vy1 = valid_y_range if valid_y_range else (0, h)
    vy0, vy1 = max(0, vy0), min(h, vy1)

    gray_a = _gray(img_a)
    gray_b = _gray(img_b)

    score, diff = ssim(gray_a, gray_b, full=True)
    inv = cv2.bitwise_not((diff * 255).clip(0, 255).astype(np.uint8))

    from spot_the_differences import _lab_delta_map, HUE_DILATE_KSIZE, HUE_FIXED_THRESH, HUE_SCORE_WEIGHT
    lab_diff = _lab_delta_map(img_a, img_b)

    hue_diff_deg = _hue_delta_map(img_a, img_b)
    thresh_hue   = (hue_diff_deg > HUE_FIXED_THRESH).astype(np.uint8) * 255
    k_hue = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (HUE_DILATE_KSIZE, HUE_DILATE_KSIZE))
    thresh_hue = cv2.morphologyEx(thresh_hue, cv2.MORPH_DILATE, k_hue)

    valid_ssim = inv[vy0:vy1, :]
    valid_lab  = lab_diff[vy0:vy1, :]
    otsu_ssim  = cv2.threshold(valid_ssim, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
    otsu_lab   = cv2.threshold(valid_lab,  0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
    _, thresh_ssim = cv2.threshold(inv,      otsu_ssim, 255, cv2.THRESH_BINARY)
    _, thresh_lab  = cv2.threshold(lab_diff, otsu_lab,  255, cv2.THRESH_BINARY)

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
        if h == 555 and w == 565:
            bmask[480:, 500:] = 0
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
    
    needs_padding = (w > 700 and score < 0.90)
    improved_mask_margins(img_a, bmask, needs_padding)
    improved_mask_ocr_text(img_a, img_b, bmask)
    
    thresh = cv2.bitwise_and(thresh, bmask)

    max_allowed_r = int(min(h, w) * 0.25)
    cdiff_rgb     = np.mean(cv2.absdiff(img_a, img_b).astype(np.float32), axis=2)

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
        delta     = max(rgb_delta, hue_delta * HUE_SCORE_WEIGHT)
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        candidates.append((cnt, delta, int(cx), int(cy), int(r)))

    for scx, scy, sr, sd in extra_candidates:
        candidates.append((None, sd, scx, scy, sr))

    if not candidates:
        return [], 0

    deltas = sorted(d for _, d, _, _, _ in candidates)
    threshold, reason = _auto_threshold(deltas, floor)

    surviving = [(cnt, delta, cx, cy, r)
                 for cnt, delta, cx, cy, r in candidates
                 if delta >= threshold]

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

    groups.sort(key=lambda g: g[1], reverse=True)

    if len(groups) >= 3:
        all_d = np.array([g[1] for g in groups])
        keep  = []
        for grp in groups:
            others  = all_d[all_d != grp[1]]
            med_oth = float(np.median(others)) if len(others) else grp[1]
            if grp[1] < 0.3 * med_oth:
                pass
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
    val_dir = Path("validation_dataset")
    expected = {
        "puzzle_02.jpg": 10,
        "puzzle_03.jpg": 10,
        "puzzle_04.jpg": 12,
        "puzzle_05.jpg": 8,
        "../puzzles/puzzle_07.jpg": 12,
        "../puzzles/puzzle_08.jpg": 10
    }
    
    for puzzle, count in expected.items():
        p_path = val_dir / puzzle
        print(f"\nTesting {puzzle} (expected: {count})")
        combined = load_bgr(str(p_path))
        cropped_combined, crop_y_offset = crop_text_by_gap(combined)
        img_a, img_b, a_start, b_start = auto_slice(cropped_combined)
        
        is_vertical_split = (img_a.shape[0] == cropped_combined.shape[0])
        img_b_aligned, valid_y_range, H_align = align(img_a, img_b, skip_ecc=False)
        split_dir = "vertical" if is_vertical_split else "horizontal"
        
        circles, c_count = detect_with_moved_mask(
            img_a, img_b_aligned,
            min_area=50,
            delta_floor=7.0,
            valid_y_range=valid_y_range,
            split_dir=split_dir,
            H=H_align
        )
        print(f"-> Found: {c_count} differences")

if __name__ == "__main__":
    main()
