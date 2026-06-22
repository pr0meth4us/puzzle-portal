import sys
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import spot_the_differences as std

def main():
    combined = std.load_bgr("puzzles/puzzle_07.jpg")
    h_orig, w_orig = combined.shape[:2]
    
    cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
    img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
    is_vertical_split = (img_a.shape[0] == cropped_combined.shape[0])
    
    img_b_aligned, valid_y_range, H_align = std.align(img_a, img_b, skip_ecc=False)
    split_dir = "vertical" if is_vertical_split else "horizontal"
    
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
    
    _, thresh_ssim = cv2.threshold(inv,      otsu_ssim, 255, cv2.THRESH_BINARY)
    _, thresh_lab  = cv2.threshold(lab_diff, otsu_lab,  255, cv2.THRESH_BINARY)
    
    thresh = cv2.bitwise_or(thresh_ssim, thresh_lab)
    thresh = cv2.bitwise_or(thresh,      thresh_hue)
    
    # Boundary mask
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
    
    # Check SIFT alignment quality and apply edge masking if needed
    edge_k = 0 # Disable Canny edge masking
    if edge_k > 0:
        gray_a_canny = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        edges_a = cv2.Canny(gray_a_canny, 50, 150)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_k, edge_k))
        dilated_edges = cv2.dilate(edges_a, kernel_edge)
        bmask = cv2.bitwise_and(bmask, cv2.bitwise_not(dilated_edges))
    
    thresh_masked = cv2.bitwise_and(thresh, bmask)
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh_masked = cv2.morphologyEx(thresh_masked, cv2.MORPH_CLOSE, k9)
    thresh_masked = cv2.morphologyEx(thresh_masked, cv2.MORPH_OPEN,  k5)
    thresh_final = cv2.bitwise_and(thresh_masked, bmask)
    
    max_allowed_r = 70
    cdiff_rgb = np.mean(cv2.absdiff(img_a, img_b_aligned).astype(np.float32), axis=2)
    
    pre_cnts, _ = cv2.findContours(thresh_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
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
    for cnt in contours:
        area = cv2.contourArea(cnt)
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
        rgb_delta = cv2.mean(cdiff_rgb, mask=m)[0]
        hue_delta = cv2.mean(hue_diff_deg, mask=m)[0]
        delta = max(rgb_delta, hue_delta * std.HUE_SCORE_WEIGHT)
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        candidates.append((cnt, delta, int(cx), int(cy), int(r), area))
        
    for scx, scy, sr, sd in extra_candidates:
        candidates.append((None, sd, scx, scy, sr, 0))
        
    print(f"Total candidates found: {len(candidates)}")
    print("Candidates details (sorted by delta desc):")
    candidates.sort(key=lambda x: x[1], reverse=True)
    for idx, (cnt, delta, cx, cy, r, area) in enumerate(candidates):
        print(f"Cand {idx:2d}: ({cx:4d}, {cy:4d}), delta={delta:5.2f}, r={r:3d}, area={area:6.1f}")

if __name__ == "__main__":
    main()
