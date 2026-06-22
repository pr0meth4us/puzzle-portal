import sys
from pathlib import Path
sys.path.append("/Users/nicksng/code/puzzle-portal")
import spot_the_difference.spot_the_differences as std
import cv2
import numpy as np
from PIL import Image

img_path = "spot_the_difference/puzzles/puzzle_08.jpg"
combined = std.load_bgr(img_path)
cropped, crop_y_offset = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped)
img_b_aligned, valid_y_range, H_align, valid_mask = std.align(img_a, img_b, skip_ecc=False)

circles, count = std.detect(img_a, img_b_aligned,
                            min_area=50,
                            delta_floor=12.0,
                            valid_y_range=valid_y_range,
                            split_dir="horizontal",
                            H=H_align,
                            edge_mask_ksize=5,
                            merge_radius_override=40,
                            valid_mask=valid_mask)

h, w = img_a.shape[:2]
diff = cv2.absdiff(img_a, img_b_aligned)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

b_slice = combined[b_start:b_start + h, :]
panel_pil = Image.fromarray(cv2.cvtColor(b_slice, cv2.COLOR_BGR2RGB))
b_slice_cv = cv2.cvtColor(np.array(panel_pil), cv2.COLOR_RGB2BGR)
pw, ph = panel_pil.size

bmask = np.ones((h, w), dtype=np.uint8) * 255
bmask[:12, :] = 0
bmask[h-12:, :] = 0
bmask[:, :12] = 0
bmask[:, w-12:] = 0

if valid_mask is not None:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_valid = cv2.erode(valid_mask, kernel)
    bmask = cv2.bitwise_and(bmask, eroded_valid)

gray_diff = cv2.bitwise_and(gray_diff, bmask)
H_inv = np.linalg.inv(H_align) if H_align is not None else None

for idx, (cx, cy, r) in enumerate(circles):
    cx, cy = int(cx), int(cy)
    r_size = max(40, int(r * 1.5))
    y1, y2 = max(0, cy - r_size), min(h, cy + r_size)
    x1, x2 = max(0, cx - r_size), min(w, cx + r_size)
    
    roi = gray_diff[y1:y2, x1:x2].copy()
    roi[0:2, :] = 0
    roi[-2:, :] = 0
    roi[:, 0:2] = 0
    roi[:, -2:] = 0
    
    max_val = np.max(roi)
    thresh_val = max(8, int(max_val * 0.25))
    _, th = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k_close)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_to_draw = []
    if cnts:
        for c in cnts:
            if cv2.contourArea(c) >= 4:
                contour_shifted = c + np.array([x1, y1])
                if H_inv is not None:
                    pts_pts = contour_shifted.astype(np.float32).reshape(-1, 1, 2)
                    warped_pts = cv2.perspectiveTransform(pts_pts, H_inv).reshape(-1, 2)
                    contour_draw = warped_pts.astype(np.int32)
                else:
                    contour_draw = contour_shifted.astype(np.int32)
                contours_to_draw.append(contour_draw.reshape(-1, 1, 2))
                
    print(f"Circle {idx+1}: center=({cx}, {cy}), r={r}, max_val={max_val}, thresh_val={thresh_val}, cnts={len(cnts)}, contours_to_draw={len(contours_to_draw)}")
