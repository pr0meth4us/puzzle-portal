import cv2
import numpy as np
import sys

sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

circles = [
    (46, 458, 28),    # 10. river patch left (book)
    (45, 539, 28),    # 11. rock left
]

combined = std.load_bgr("puzzles/puzzle_07.jpg")
cropped, y_off = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped)
img_b_aligned, valid_y, H = std.align(img_a, img_b, skip_ecc=False)

h, w = img_a.shape[:2]
diff = cv2.absdiff(img_a, img_b_aligned)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

bmask = np.ones((h, w), dtype=np.uint8) * 255
bmask[:, :40] = 0
bmask[:, w - 32:] = 0
bmask[:32, :] = 0
bmask[h - 32:, :] = 0
bgr_sum = np.sum(img_b_aligned, axis=2)
bmask[bgr_sum < 15] = 0

gray_diff_masked = cv2.bitwise_and(gray_diff, bmask)

for idx, (cx, cy, r) in enumerate(circles):
    r_size = 40
    y1, y2 = max(0, cy - r_size), min(h, cy + r_size)
    x1, x2 = max(0, cx - r_size), min(w, cx + r_size)
    
    roi_unmasked = gray_diff[y1:y2, x1:x2]
    roi_masked = gray_diff_masked[y1:y2, x1:x2]
    
    _, th_unmasked = cv2.threshold(roi_unmasked, 15, 255, cv2.THRESH_BINARY)
    _, th_masked = cv2.threshold(roi_masked, 8, 255, cv2.THRESH_BINARY)
    
    cnts_unmasked, _ = cv2.findContours(th_unmasked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_masked, _ = cv2.findContours(th_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\nDifference {idx+10}: cx={cx}, cy={cy}")
    print(f"  ROI bounds: x=({x1}, {x2}), y=({y1}, {y2})")
    print(f"  Unmasked ROI non-zero pixels: {np.count_nonzero(roi_unmasked)}")
    print(f"  Masked ROI non-zero pixels: {np.count_nonzero(roi_masked)}")
    print(f"  Unmasked contours count: {len(cnts_unmasked)}")
    print(f"  Masked contours count: {len(cnts_masked)}")
    
    for c_idx, c in enumerate(cnts_masked):
        area = cv2.contourArea(c)
        pts = c.reshape(-1, 2)
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        print(f"    Masked contour {c_idx+1}: area={area}, bbox=({min_x+x1}, {min_y+y1}) -> ({max_x+x1}, {max_y+y1})")
