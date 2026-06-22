import cv2
import numpy as np
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

truth_rough = [
    (453.0, 210.4),   # 1. dam top-left rock
    (617.2, 503.4),   # 2. dam middle
    (211.5, 564.6),   # 3. water left
    (408.4, 791.5),   # 4. dam lower
    (953.1, 811.8),   # 5. dam right
    (387.2, 857.2),   # 6. dam lower-left
    (632.7, 870.7),   # 7. dam lower-middle
    (295.9, 899.5),   # 8. dam rock
    (809.1, 933.0),   # 9. dam lower-right
    (45.9, 457.7),    # 10. river patch left
    (44.7, 538.5),    # 11. rock left
    (1076.7, 440.1)   # 12. figure label at right edge
]

puz = cv2.imread("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(puz)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
img_b_aligned, valid_y_range, H_align = std.align(img_a, img_b, skip_ecc=False)

cdiff_rgb = np.mean(cv2.absdiff(img_a, img_b_aligned).astype(np.float32), axis=2)

print("Refined Ground Truth Coordinates:")
for i, (tx, ty) in enumerate(truth_rough):
    tx, ty = int(tx), int(ty)
    
    # Extract ROI around rough coordinate
    r = 60
    y1, y2 = max(0, ty - r), min(img_a.shape[0], ty + r)
    x1, x2 = max(0, tx - r), min(img_a.shape[1], tx + r)
    
    roi = cdiff_rgb[y1:y2, x1:x2].copy()
    
    # Find local maximum or centroid of the highest diff blob in ROI
    _, th = cv2.threshold(roi, 10.0, 255, cv2.THRESH_BINARY)
    th = th.astype(np.uint8)
    
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_cx, best_cy = tx, ty
    max_val = -1
    
    for c in cnts:
        mask = np.zeros(roi.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
        val = cv2.mean(roi, mask=mask)[0]
        
        # Calculate centroid of this contour in ROI
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx_roi = M["m10"] / M["m00"]
            cy_roi = M["m01"] / M["m00"]
            cx_full = x1 + cx_roi
            cy_full = y1 + cy_roi
            
            # We want the highest difference value that is close to center
            dist = np.hypot(cx_full - tx, cy_full - ty)
            score = val / (1.0 + 0.05 * dist) # penalize far away candidates slightly
            if score > max_val:
                max_val = score
                best_cx = cx_full
                best_cy = cy_full
                
    print(f"  GT {i+1}: Center=({best_cx:.2f}, {best_cy:.2f}), Delta={max_val:.2f}")
