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
    (1036.0, 430.0)   # 12. figure label at right edge
]

puz = cv2.imread("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(puz)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
img_b_aligned, _, _ = std.align(img_a, img_b, skip_ecc=False)

rows = []
for i, (tx, ty) in enumerate(truth_actual):
    tx, ty = int(tx), int(ty)
    # Crop 100x100
    y1, y2 = max(0, ty - 50), min(img_a.shape[0], ty + 50)
    x1, x2 = max(0, tx - 50), min(img_a.shape[1], tx + 50)
    
    crop_a = img_a[y1:y2, x1:x2].copy()
    crop_b = img_b_aligned[y1:y2, x1:x2].copy()
    
    # Pad to 100x100 if at borders
    if crop_a.shape != (100, 100, 3):
        crop_a = cv2.resize(crop_a, (100, 100))
        crop_b = cv2.resize(crop_b, (100, 100))
        
    # Draw label on crop_a
    cv2.putText(crop_a, f"GT {i+1}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Concatenate side-by-side
    pair = cv2.hconcat([crop_a, crop_b])
    rows.append(pair)

# Arrange in 4 columns, 3 rows of pairs
# Each pair is 200x100
grid_rows = []
for r in range(3):
    grid_row = cv2.hconcat(rows[r*4:(r+1)*4])
    grid_rows.append(grid_row)
grid = cv2.vconcat(grid_rows)

cv2.imwrite("/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/all_gts.png", grid)
print("Saved GT comparison grid to artifacts/all_gts.png")
