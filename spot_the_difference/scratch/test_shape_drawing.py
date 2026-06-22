import cv2
import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont
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

combined = std.load_bgr("puzzles/puzzle_07.jpg")
cropped, y_off = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped)
img_b_aligned, valid_y, H = std.align(img_a, img_b, skip_ecc=False)

h, w = img_a.shape[:2]
H_inv = np.linalg.inv(H) if H is not None else None

# Calculate the difference map
diff = cv2.absdiff(img_a, img_b_aligned)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Prepare PIL image of B panel to draw on
# We extract the B slice from the original combined image
b_slice = combined[b_start:b_start + h, :]
pil_b = Image.fromarray(cv2.cvtColor(b_slice, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pil_b)

# Select a bold font for numbers
font_size = 28
try:
    font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", font_size)
except Exception:
    font = ImageFont.load_default()

color = (244, 9, 192) # Vibrant pink/magenta
count = len(truth_refined)

for idx, (cx, cy) in enumerate(truth_refined):
    cx, cy = int(cx), int(cy)
    
    # 1. Define ROI (80x80)
    r_size = 40
    y1, y2 = max(0, cy - r_size), min(h, cy + r_size)
    x1, x2 = max(0, cx - r_size), min(w, cx + r_size)
    
    roi = gray_diff[y1:y2, x1:x2].copy()
    
    # 2. Threshold locally to get the exact difference shape
    _, th = cv2.threshold(roi, 15, 255, cv2.THRESH_BINARY)
    
    # 3. Find contours
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        continue
        
    # Get the largest contour in the ROI
    largest_cnt = max(cnts, key=cv2.contourArea)
    
    # Shift contour points to full img_a coordinate space
    contour_shifted = largest_cnt + np.array([x1, y1])
    
    # 4. Warp contour points to img_b space using H_inv
    if H_inv is not None:
        pts = contour_shifted.astype(np.float32).reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
        contour_draw = warped_pts.astype(np.int32)
    else:
        contour_draw = contour_shifted.astype(np.int32)
        
    # 5. Draw the contour shape on B panel
    b_slice_cv = cv2.cvtColor(np.array(pil_b), cv2.COLOR_RGB2BGR)
    cv2.polylines(b_slice_cv, [contour_draw], isClosed=True, color=color[::-1], thickness=3)
    pil_b = Image.fromarray(cv2.cvtColor(b_slice_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_b)
    
    # 6. Put the number next to the contour bounding box
    x_coords = contour_draw[:, 0]
    y_coords = contour_draw[:, 1]
    x_min, y_min = int(np.min(x_coords)), int(np.min(y_coords))
    x_max, y_max = int(np.max(x_coords)), int(np.max(y_coords))
    
    # Place number slightly above/left of the bounding box
    num_str = str(idx + 1)
    tx = max(5, x_min - 25)
    ty = max(5, y_min - 25)
    
    # Draw number with a small dark background shadow for visibility
    draw.text((tx+1, ty+1), num_str, font=font, fill=(0, 0, 0))
    draw.text((tx, ty), num_str, font=font, fill=(255, 255, 255))

# Re-assemble the combined output stacked image
BH = 80
oh, ow = combined.shape[:2]
pil_a = Image.fromarray(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
base = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
canvas = Image.new("RGB", (ow, BH + oh), (30, 30, 50))
canvas.paste(base, (0, BH))
canvas.paste(pil_a, (0, BH + a_start))
canvas.paste(pil_b, (0, BH + b_start))
canvas.paste(std.make_khmer_banner(ow, count), (0, 0))

canvas = std.add_watermark(canvas)
canvas.save("/Users/nicksng/code/puzzle-portal/spot_the_difference/results/res_puzzle_07.png", quality=95)
print("Saved contour-drawn result to results/res_puzzle_07.png")
