import cv2
import numpy as np

img = cv2.imread("spot_the_difference/results/res_puzzle_08.png")
if img is None:
    print("Could not read res_puzzle_08.png")
    exit()

# We don't know the exact run color since it is randomized, but we know the background is mostly not green/cyan,
# or we can find the most common color in the drawing if we search for non-white/non-black non-background pixels,
# or we can find pixels that form thick lines.
# Actually, let's convert to HSV and find non-grey pixels that form contours in panel B (bottom half).
h, w = img.shape[:2]
panel_b = img[h//2:]

hsv = cv2.cvtColor(panel_b, cv2.COLOR_BGR2HSV)
# The drawing color is randomized, but usually highly saturated (S > 150) and bright (V > 150)
mask = cv2.inRange(hsv, np.array([0, 150, 150]), np.array([180, 255, 255]))

cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(cnts)} saturated contours in bottom panel:")

for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    if area < 50:
        continue
    peri = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / (peri ** 2) if peri > 0 else 0
    x, y, w_c, h_c = cv2.boundingRect(c)
    print(f"  Shape {i+1}: area={area:.1f}, circularity={circularity:.3f}, bounding box=[{x}, {y}, {w_c}, {h_c}]")
