import cv2
import numpy as np
import os
import sys

# Coordinates of the 12 differences
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

img = cv2.imread("results/res_puzzle_07_dynamic.png")
h, w = img.shape[:2]

os.makedirs("/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/dynamic_crops", exist_ok=True)

sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

combined = cv2.imread("puzzles/puzzle_07.jpg")
cropped, y_off = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped)

H = np.array([
    [1.00062325e+00, 3.49079361e-04, -3.76632420e+01],
    [-3.21045233e-04, 1.00057404e+00, 1.94042299e-01],
    [-4.51268307e-07, 7.82869168e-07, 1.00000000e+00]
]) # H from SIFT
H_inv = np.linalg.inv(H)

for idx, (cx, cy, r) in enumerate(circles):
    pts = np.float32([[cx, cy]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
    mcx, mcy = int(mapped[0][0]), int(mapped[0][1])
    
    panel_y = 80 + b_start + mcy
    panel_x = mcx
    
    # Crop 180x180 box to capture the larger number as well
    y1, y2 = max(0, panel_y - 90), min(h, panel_y + 90)
    x1, x2 = max(0, panel_x - 90), min(w, panel_x + 90)
    
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(f"/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/dynamic_crops/diff_{idx+1}.png", crop)
    print(f"Saved diff_{idx+1}.png")
