import cv2
import numpy as np
import sys
import os
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

drawn_items = []

for idx, (cx, cy, r) in enumerate(circles):
    cx, cy = int(cx), int(cy)
    
    r_size = max(40, int(r * 1.5))
    y1, y2 = max(0, cy - r_size), min(h, cy + r_size)
    x1, x2 = max(0, cx - r_size), min(w, cx + r_size)
    
    roi = gray_diff[y1:y2, x1:x2].copy()
    
    # Zero out the border of the ROI to prevent boundary hugging
    roi[0:2, :] = 0
    roi[-2:, :] = 0
    roi[:, 0:2] = 0
    roi[:, -2:] = 0
    
    max_val = np.max(roi)
    # Dynamic threshold based on max value in ROI
    thresh_val = max(8, int(max_val * 0.25))
    
    _, th = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Light closing to group nearby components slightly
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k_close)
    
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_to_draw = []
    if cnts:
        # Keep contours with area >= 4 pixels to filter tiny noise
        for c in cnts:
            if cv2.contourArea(c) >= 4:
                # Shift contour points to full img_a space
                contour_shifted = c + np.array([x1, y1])
                # Warp to B space
                if H_inv is not None:
                    pts_pts = contour_shifted.astype(np.float32).reshape(-1, 1, 2)
                    warped_pts = cv2.perspectiveTransform(pts_pts, H_inv).reshape(-1, 2)
                    contour_draw = warped_pts.astype(np.int32)
                else:
                    contour_draw = contour_shifted.astype(np.int32)
                contours_to_draw.append(contour_draw.reshape(-1, 1, 2))
                
    if contours_to_draw:
        for c_draw in contours_to_draw:
            cv2.polylines(b_slice_cv, [c_draw], isClosed=True, color=(192, 9, 244), thickness=3)
        drawn_items.append(("contours", contours_to_draw, warped_circles[idx]))
    else:
        fallback_cx, fallback_cy, fallback_r = warped_circles[idx]
        cv2.circle(b_slice_cv, (fallback_cx, fallback_cy), fallback_r, (192, 9, 244), thickness=3)
        drawn_items.append(("circle", [], warped_circles[idx]))

pil_b = Image.fromarray(cv2.cvtColor(b_slice_cv, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pil_b)

font_size = 54  # Larger font size
font = std._load_font(std._LATIN_FONT, font_size)

for idx, (dtype, cnts_draw, (fcx, fcy, fr)) in enumerate(drawn_items):
    num_str = str(idx + 1)
    
    if dtype == "contours" and cnts_draw:
        # Find the overall bounding box of all contours for this difference
        all_pts = np.vstack([c.reshape(-1, 2) for c in cnts_draw])
        x_min = int(np.min(all_pts[:, 0]))
        y_min = int(np.min(all_pts[:, 1]))
        tx = max(5, x_min - 45)
        ty = max(5, y_min - 45)
    else:
        tx = max(5, fcx - fr - 30)
        ty = max(5, fcy - fr - 30)
        
    # Draw number with thick black outline for maximum contrast
    draw.text((tx, ty), num_str, font=font, fill=(255, 255, 255), stroke_width=4, stroke_fill=(0, 0, 0))

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
canvas.save("/Users/nicksng/code/puzzle-portal/spot_the_difference/results/res_puzzle_07_dynamic.png", quality=95)
print("Saved dynamic result to results/res_puzzle_07_dynamic.png")
