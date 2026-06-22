import cv2
import numpy as np

img_puz = cv2.imread("validation_dataset/puzzle_05.jpg")
img_ans = cv2.imread("correct_answers/answer_05.jpg")

if img_puz is None or img_ans is None:
    print("Could not read images.")
    exit()

# SIFT matching to find homography between puzzle_05 and answer_05
sift = cv2.SIFT_create()
kp_p, des_p = sift.detectAndCompute(img_puz, None)
kp_a, des_a = sift.detectAndCompute(img_ans, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des_p, des_a)
pts_p = np.float32([kp_p[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts_a = np.float32([kp_a[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

H_p_to_a, mask = cv2.findHomography(pts_p, pts_a, cv2.RANSAC, 5.0)
H_a_to_p = np.linalg.inv(H_p_to_a)

# Let's find red circles in answer_05.jpg
hsv = cv2.cvtColor(img_ans, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
mask_red = mask1 | mask2
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, k)

cnts, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
gt_circles_in_ans = []
for c in cnts:
    area = cv2.contourArea(c)
    if area < 50:
        continue
    (cx, cy), r = cv2.minEnclosingCircle(c)
    gt_circles_in_ans.append((cx, cy, r))

# Map gt circles back to puzzle_05 coordinates
gt_circles_in_puz = []
for cx, cy, r in gt_circles_in_ans:
    pt = np.float32([cx, cy]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pt, H_a_to_p).reshape(2)
    # The radius scale factor is roughly the average scaling
    scale = np.sqrt(H_a_to_p[0,0]**2 + H_a_to_p[0,1]**2)
    gt_circles_in_puz.append((mapped[0], mapped[1], r * scale))

print(f"Found {len(gt_circles_in_puz)} Ground Truth circles in puzzle_05 space:")
for i, (cx, cy, r) in enumerate(gt_circles_in_puz):
    # Crop coordinates in puzzle_05 are relative to top/bottom panels
    # Let's determine if it is in Panel A (top) or Panel B (bottom)
    # puzzle_05 height is 1144, each panel starts after crop_text_by_gap
    # Let's print the raw coordinates
    print(f"  GT {i+1}: Center=({cx:.1f}, {cy:.1f}), r={r:.1f}")

# Detected circles with mask-roi
detected_with_mask = [
    (459, 292), (78, 511), (534, 379), (248, 314), (551, 423), (257, 202), (38, 179), (426, 158)
]

# Detected circles without mask-roi (which has the extra circle at 543, 515)
detected_no_mask = [
    (459, 292), (78, 511), (534, 379), (248, 314), (551, 423), (257, 202), (38, 179), (426, 158), (543, 515)
]

import sys
from pathlib import Path
SCRIPT_DIR = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
import spot_the_differences as std
cropped, crop_y = std.crop_text_by_gap(img_puz)
img_a, img_b, a_start, b_start = std.auto_slice(cropped)
print(f"\nPanel A y range in puzzle_05: {crop_y + a_start} to {crop_y + a_start + img_a.shape[0]}")
print(f"Panel B y range in puzzle_05: {crop_y + b_start} to {crop_y + b_start + img_b.shape[0]}")

panel_h = img_a.shape[0]
y_offset = crop_y + a_start

print("\nMatching DETECTED WITH MASK against GT:")
matched_gt = set()
for idx, (dcx, dcy) in enumerate(detected_with_mask):
    # Map dcx, dcy to full puzzle_05 coordinate space
    # (dcx, dcy) is in Panel A space, so x_full = dcx, y_full = y_offset + dcy
    fx, fy = dcx, y_offset + dcy
    closest_dist = float('inf')
    closest_gt = -1
    for i, (gcx, gcy, gr) in enumerate(gt_circles_in_puz):
        dist = np.sqrt((fx - gcx)**2 + (fy - gcy)**2)
        if dist < closest_dist:
            closest_dist = dist
            closest_gt = i
    if closest_dist < 40:
        matched_gt.add(closest_gt)
        print(f"  Detected {idx+1} at ({dcx}, {dcy}) -> MATCHED GT {closest_gt+1} (dist={closest_dist:.1f})")
    else:
        print(f"  Detected {idx+1} at ({dcx}, {dcy}) -> FALSE POSITIVE (closest GT {closest_gt+1} dist={closest_dist:.1f})")

print(f"Total matched GTs with mask: {len(matched_gt)}")

print("\nMatching DETECTED WITHOUT MASK against GT:")
matched_gt_no = set()
for idx, (dcx, dcy) in enumerate(detected_no_mask):
    fx, fy = dcx, y_offset + dcy
    closest_dist = float('inf')
    closest_gt = -1
    for i, (gcx, gcy, gr) in enumerate(gt_circles_in_puz):
        dist = np.sqrt((fx - gcx)**2 + (fy - gcy)**2)
        if dist < closest_dist:
            closest_dist = dist
            closest_gt = i
    if closest_dist < 40:
        matched_gt_no.add(closest_gt)
        print(f"  Detected {idx+1} at ({dcx}, {dcy}) -> MATCHED GT {closest_gt+1} (dist={closest_dist:.1f})")
    else:
        print(f"  Detected {idx+1} at ({dcx}, {dcy}) -> FALSE POSITIVE (closest GT {closest_gt+1} dist={closest_dist:.1f})")

print(f"Total matched GTs without mask: {len(matched_gt_no)}")
print("Missed GTs without mask:")
for i in range(len(gt_circles_in_puz)):
    if i not in matched_gt_no:
        print(f"  GT {i+1} at {gt_circles_in_puz[i]}")
