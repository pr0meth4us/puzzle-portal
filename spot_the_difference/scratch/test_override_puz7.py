import cv2
import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont

sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

# Define correct ground truth coordinates
circles = [
    (453, 210, 28),   # 1. dam top-left rock
    (617, 503, 28),   # 2. dam middle
    (211, 565, 28),   # 3. water left
    (408, 792, 28),   # 4. dam lower
    (953, 812, 28),   # 5. dam right
    (387, 857, 28),   # 6. dam lower-left
    (633, 871, 28),   # 7. dam lower-middle
    (296, 900, 28),   # 8. dam rock
    (809, 933, 28),   # 9. dam lower-right
    (46, 458, 28),    # 10. river patch left (book)
    (45, 539, 28),    # 11. rock left
    (1095, 449, 28)   # 12. figure label at right edge
]

combined = std.load_bgr("puzzles/puzzle_07.jpg")
cropped, y_off = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped)
img_b_aligned, valid_y, H = std.align(img_a, img_b, skip_ecc=False)

h, w = img_a.shape[:2]
H_inv = np.linalg.inv(H) if H is not None else None

# Difference map
diff = cv2.absdiff(img_a, img_b_aligned)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Apply border and empty warp masking to gray_diff
bmask = np.ones((h, w), dtype=np.uint8) * 255
bmask[:, :40] = 0
bmask[:, w - 40:] = 0
bmask[:32, :] = 0
bmask[h - 32:, :] = 0
bgr_sum = np.sum(img_b_aligned, axis=2)
bmask[bgr_sum < 15] = 0

gray_diff = cv2.bitwise_and(gray_diff, bmask)

# Prepare PIL image of B panel to draw on
ph = img_a.shape[0]
b_slice = combined[b_start:b_start + ph, :]
pil_b = Image.fromarray(cv2.cvtColor(b_slice, cv2.COLOR_BGR2RGB))
b_slice_cv = cv2.cvtColor(np.array(pil_b), cv2.COLOR_RGB2BGR)

pw, ph = pil_b.size
warped_circles = []
if H_inv is not None:
    pts = np.float32([[cx, cy] for cx, cy, r in circles]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
    warped_circles = [(int(np.clip(mx, 0, pw-1)), int(np.clip(my, 0, ph-1)), r)
                      for (mx, my), (_, _, r) in zip(mapped, circles)]
else:
    warped_circles = [(cx, cy, r) for cx, cy, r in circles]

drawn_contours = []

for idx, (cx, cy, r) in enumerate(circles):
    cx, cy = int(cx), int(cy)
    
    r_size = max(40, int(r * 1.5))
    y1, y2 = max(0, cy - r_size), min(h, cy + r_size)
    x1, x2 = max(0, cx - r_size), min(w, cx + r_size)
    
    roi = gray_diff[y1:y2, x1:x2].copy()
    
    # Lower threshold to 8 to capture low-contrast book/rock differences
    _, th = cv2.threshold(roi, 8, 255, cv2.THRESH_BINARY)
    
    # Apply local morphological closing to merge close components
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k_close)
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.dilate(th, k_dilate)
    
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_draw = None
    if cnts:
        roi_cx = cx - x1
        roi_cy = cy - y1
        best_cnt = None
        min_dist = float('inf')
        
        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] > 0:
                ccx = M["m10"] / M["m00"]
                ccy = M["m01"] / M["m00"]
            else:
                pts_mean = np.mean(c.reshape(-1, 2), axis=0)
                ccx, ccy = pts_mean[0], pts_mean[1]
                
            dist = np.hypot(ccx - roi_cx, ccy - roi_cy)
            if dist < min_dist:
                min_dist = dist
                best_cnt = c
        
        if best_cnt is not None and min_dist < r_size * 1.2:
            contour_shifted = best_cnt + np.array([x1, y1])
            if H_inv is not None:
                pts_pts = contour_shifted.astype(np.float32).reshape(-1, 1, 2)
                warped_pts = cv2.perspectiveTransform(pts_pts, H_inv).reshape(-1, 2)
                contour_draw = warped_pts.astype(np.int32)
            else:
                contour_draw = contour_shifted.astype(np.int32)
                
    if contour_draw is not None:
        contour_draw = contour_draw.reshape(-1, 1, 2)
        cv2.polylines(b_slice_cv, [contour_draw], isClosed=True, color=(192, 9, 244), thickness=3) # Vibrant pink/magenta BGR is (192, 9, 244)
        drawn_contours.append(contour_draw)
    else:
        fallback_cx, fallback_cy, fallback_r = warped_circles[idx]
        cv2.circle(b_slice_cv, (fallback_cx, fallback_cy), fallback_r, (192, 9, 244), thickness=3)
        drawn_contours.append(None)

pil_b = Image.fromarray(cv2.cvtColor(b_slice_cv, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pil_b)

font_size = 38
font = std._load_font(std._LATIN_FONT, font_size)

for idx, contour_draw in enumerate(drawn_contours):
    num_str = str(idx + 1)
    if contour_draw is not None:
        contour_draw_2d = contour_draw.reshape(-1, 2)
        x_coords = contour_draw_2d[:, 0]
        y_coords = contour_draw_2d[:, 1]
        x_min, y_min = int(np.min(x_coords)), int(np.min(y_coords))
        tx = max(5, x_min - 35)
        ty = max(5, y_min - 35)
    else:
        fallback_cx, fallback_cy, fallback_r = warped_circles[idx]
        tx = max(5, fallback_cx - fallback_r - 20)
        ty = max(5, fallback_cy - fallback_r - 20)
        
    draw.text((tx+1, ty+1), num_str, font=font, fill=(0, 0, 0))
    draw.text((tx, ty), num_str, font=font, fill=(255, 255, 255))

BH = 80
oh, ow = combined.shape[:2]
pil_a = Image.fromarray(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
base = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
canvas = Image.new("RGB", (ow, BH + oh), (30, 30, 50))
canvas.paste(base, (0, BH))
canvas.paste(pil_a, (0, BH + a_start))
canvas.paste(pil_b, (0, BH + b_start))
canvas.paste(std.make_khmer_banner(ow, 12), (0, 0))
canvas = std.add_watermark(canvas)
canvas.save("/Users/nicksng/code/puzzle-portal/spot_the_difference/results/res_puzzle_07_test.png", quality=95)
print("Saved test result to results/res_puzzle_07_test.png")
