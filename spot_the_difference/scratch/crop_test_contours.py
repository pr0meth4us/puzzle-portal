import cv2
import numpy as np
import os

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

# Read the test result image
img = cv2.imread("results/res_puzzle_07_test.png")
h, w = img.shape[:2]

# Let's crop each difference and save it to artifacts for inspection
os.makedirs("/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/test_crops", exist_ok=True)

# The test result has a Khmer banner at the top (height 80).
# The B panel starts at b_start. Let's see the b_start.
# The horizontal auto slice slices a combined image (1006 x 1152).
# Let's load the original combined to know the slice boundaries.
combined = cv2.imread("puzzles/puzzle_07.jpg")
cropped, y_off = combined.copy(), 0  # Assuming crop_text_by_gap didn't change heights much, let's do auto_slice
# Let's run std.crop_text_by_gap to match the exact slice
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std
cropped, y_off = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped)

# Let's crop from the bottom panel of res_puzzle_07_test.png
# In the test result, B panel starts at y = 80 + b_start
for idx, (cx, cy, r) in enumerate(circles):
    # Map (cx, cy) from B aligned space to the output image's B panel
    # Wait, the B panel in res_puzzle_07_test.png was drawn directly on the cropped slice (or aligned slice?)
    # In test_override_puz7.py:
    # b_slice = combined[b_start:b_start + ph, :] -> this is the original B panel!
    # And we draw on b_slice using the warped_pts / warped_circles
    # And then we paste it at BH + b_start
    # So the coordinates on the B panel are warped by H_inv!
    # Let's just find where the drawn contour is by looking around the warped coordinates.
    H = np.array([
        [1.00062325e+00, 3.49079361e-04, -3.76632420e+01],
        [-3.21045233e-04, 1.00057404e+00, 1.94042299e-01],
        [-4.51268307e-07, 7.82869168e-07, 1.00000000e+00]
    ]) # H from SIFT
    H_inv = np.linalg.inv(H)
    pts = np.float32([[cx, cy]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
    mcx, mcy = int(mapped[0][0]), int(mapped[0][1])
    
    # We crop around (mcx, mcy) in the B panel of the test output image
    # The B panel starts at 80 + b_start.
    panel_y = 80 + b_start + mcy
    panel_x = mcx
    
    # Crop 120x120 box
    y1, y2 = max(0, panel_y - 80), min(h, panel_y + 80)
    x1, x2 = max(0, panel_x - 80), min(w, panel_x + 80)
    
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(f"/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/test_crops/diff_{idx+1}.png", crop)
    print(f"Saved diff_{idx+1}.png at center ({panel_x}, {panel_y})")
