import cv2
import numpy as np
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

truth_actual = [
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
hue_diff_deg = std._hue_delta_map(img_a, img_b_aligned)

print("Ground Truth Deltas:")
for i, (tx, ty) in enumerate(truth_actual):
    tx, ty = int(tx), int(ty)
    # Mask of radius 15 around GT
    m = np.zeros(img_a.shape[:2], dtype=np.uint8)
    cv2.circle(m, (tx, ty), 15, 255, -1)
    
    rgb_delta = cv2.mean(cdiff_rgb, mask=m)[0]
    hue_delta = cv2.mean(hue_diff_deg, mask=m)[0]
    delta = max(rgb_delta, hue_delta * std.HUE_SCORE_WEIGHT)
    print(f"  GT {i+1} at ({tx}, {ty}): RGB_delta={rgb_delta:.2f}, Hue_delta={hue_delta:.2f}, Final_delta={delta:.2f}")
