import cv2
import numpy as np

img = cv2.imread("correct_answers/answer_01.jpg")
if img is None:
    print("Could not read image")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Since circles are hand-drawn black/dark lines on a light background, we can detect them by looking for dark contours
# that have circular shape (high circularity / compact shape).
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

# Find contours
cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

circles = []
if hierarchy is not None:
    hierarchy = hierarchy[0]
    for i, c in enumerate(cnts):
        # We look for contours that have an inner child (hollow circle/outline)
        # or contours with a certain perimeter/area ratio (circularity)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if area < 50 or perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Hand-drawn circles might not be perfectly circular, so circularity > 0.4
        if circularity > 0.4:
            (cx, cy), r = cv2.minEnclosingCircle(c)
            # Check if this contour is a child or parent
            circles.append((cx, cy, r, circularity, area))

print(f"Found {len(circles)} candidate circular contours in answer_01.jpg:")
# Sort by area descending
circles.sort(key=lambda x: -x[4])
for i, (cx, cy, r, circ, area) in enumerate(circles[:30]):
    print(f"  Circle {i+1}: center=({cx:.1f}, {cy:.1f}), r={r:.1f}, circularity={circ:.2f}, area={area:.1f}")
