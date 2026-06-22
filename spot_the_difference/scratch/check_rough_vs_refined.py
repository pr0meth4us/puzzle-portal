import cv2
import os
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

combined = std.load_bgr("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
img_b_aligned, _, H = std.align(img_a, img_b, skip_ecc=False)

os.makedirs("scratch/crops", exist_ok=True)

for i, (tx, ty) in enumerate(truth_rough):
    tx, ty = int(tx), int(ty)
    r = 60
    y1, y2 = max(0, ty - r), min(img_a.shape[0], ty + r)
    x1, x2 = max(0, tx - r), min(img_a.shape[1], tx + r)
    
    crop_a = img_a[y1:y2, x1:x2]
    crop_b = img_b_aligned[y1:y2, x1:x2]
    
    # Save both crops side-by-side
    side_by_side = cv2.hconcat([crop_a, crop_b])
    cv2.imwrite(f"scratch/crops/gt_{i+1:02d}.png", side_by_side)

print("Saved crops to scratch/crops/")
