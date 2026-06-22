import cv2
import numpy as np
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

img_a = std.load_bgr("spot_the_difference/puzzles/puzzle_extra_05.jpg")
img_b = std.load_bgr("spot_the_difference/puzzles/puzzle_extra_06.jpg")

img_b_aligned, valid_y_range, H_align, valid_mask = std.align(img_a, img_b, skip_ecc=False)

# Let's run the preprocessing of detect_line:
LINE_SSIM_THRESH  = 30
LINE_MORPH_KSIZE  = 3
LINE_MIN_AREA     = 20
LINE_DELTA_FLOOR  = 5.0

h, w  = img_a.shape[:2]
img_b_aligned = cv2.resize(img_b_aligned, (w, h), interpolation=cv2.INTER_LANCZOS4)
score, diff = std.ssim(std._gray(img_a), std._gray(img_b_aligned), full=True)
inv      = cv2.bitwise_not((diff * 255).clip(0, 255).astype(np.uint8))
_, thresh = cv2.threshold(inv, LINE_SSIM_THRESH, 255, cv2.THRESH_BINARY)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (LINE_MORPH_KSIZE, LINE_MORPH_KSIZE))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k)

border = max(8, int(min(h, w) * 0.02))
bmask  = np.zeros_like(thresh)
bmask[border:h - border, border:w - border] = 255
std._mask_margins(img_a, bmask)
std._mask_ocr_text(img_a, img_b_aligned, bmask)
thresh_masked = cv2.bitwise_and(thresh, bmask)

# Find contours
contours, _ = cv2.findContours(thresh_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cdiff = np.mean(cv2.absdiff(img_a, img_b_aligned).astype(np.float32), axis=2)
ga    = std._gray(img_a).astype(np.float64)
gb    = std._gray(img_b_aligned).astype(np.float64)

# Bounding box of each contour to inspect:
print("All contours in thresh_masked:")
for idx, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    
    # Calculate delta
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
    delta = cv2.mean(cdiff, mask=mask)[0]
    
    # Calculate peak
    cx, cy, r = int(cx), int(cy), int(r)
    x1 = max(0, cx-r-5); y1 = max(0, cy-r-5)
    x2 = min(w, cx+r+5); y2 = min(h, cy+r+5)
    peak = float(np.abs(ga[y1:y2, x1:x2] - gb[y1:y2, x1:x2]).max())
    
    print(f"  Contour {idx+1}: center=({cx}, {cy}), area={area:.1f}, delta={delta:.1f}, peak={peak:.1f}")

gt_mapped = [
    (109.4, 64.1),
    (544.9, 84.8),
    (294.0, 87.6),
    (438.8, 100.0),
    (112.0, 127.5),
    (207.1, 218.5),
    (557.4, 291.7),
    (55.2, 313.8),
    (321.5, 364.9),
    (517.5, 395.2)
]

print("\nFor each Ground Truth:")
for i, (gtx, gty) in enumerate(gt_mapped):
    # Find if there is any contour nearby
    closest_cnt = -1
    closest_dist = float('inf')
    for idx, cnt in enumerate(contours):
        (cx, cy), _ = cv2.minEnclosingCircle(cnt)
        dist = np.sqrt((cx - gtx)**2 + (cy - gty)**2)
        if dist < closest_dist:
            closest_dist = dist
            closest_cnt = idx
            
    print(f"  GT {i+1} at ({gtx:.1f}, {gty:.1f}):")
    if closest_dist < 40:
        cnt = contours[closest_cnt]
        area = cv2.contourArea(cnt)
        (cx, cy), _ = cv2.minEnclosingCircle(cnt)
        print(f"    Closest Contour {closest_cnt+1} center=({cx:.1f}, {cy:.1f}), dist={closest_dist:.1f}, area={area:.1f}")
    else:
        print(f"    No contour within 40px! (Closest is {closest_dist:.1f}px away)")
