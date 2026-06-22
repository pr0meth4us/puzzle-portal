import cv2
import numpy as np

img_orig = cv2.imread("puzzles/puzzle_08.jpg")
img_res = cv2.imread("results/res_puzzle_08.png")

if img_orig is None or img_res is None:
    print("Could not read images.")
    exit()

# Since res_puzzle_08.png has a khmer banner at the top (height 80), let's crop both to the panel area.
# In res_puzzle_08.png, the panel A starts at y = 80, panel B starts at y = 80 + panel_height.
# Let's align and crop to panel B.
h_orig, w_orig = img_orig.shape[:2]
h_res, w_res = img_res.shape[:2]

# img_orig height is 2018 (2 panels of 1009).
# img_res height is 2018 + 80 = 2098.
panel_h = 1009
panel_b_res = img_res[80 + panel_h: 80 + 2 * panel_h, :]
panel_b_orig = img_orig[panel_h: 2 * panel_h, :]

# Subtract the original panel from the result panel
diff = cv2.absdiff(panel_b_res, panel_b_orig)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Threshold to find drawing strokes
_, thresh = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)

# Morphological open to remove noise
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)

cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Isolated {len(cnts)} drawn contours:")
circle_count = 0
contour_count = 0
for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    if area < 50:
        continue
    peri = cv2.arcLength(c, True)
    circ = 4 * np.pi * area / (peri ** 2) if peri > 0 else 0
    (cx, cy), r = cv2.minEnclosingCircle(c)
    
    # Check if circularity is high
    if circ > 0.85:
        shape_type = "CIRCLE"
        circle_count += 1
    else:
        shape_type = "CONTOUR"
        contour_count += 1
    print(f"  Drawn Shape {i+1}: Center=({cx:.1f}, {cy:.1f}), Radius={r:.1f}, Area={area:.1f}, Circularity={circ:.3f} -> {shape_type}")

print(f"\nSummary of drawn shapes in res_puzzle_08.png: {circle_count} CIRCLES, {contour_count} CONTOURS")
