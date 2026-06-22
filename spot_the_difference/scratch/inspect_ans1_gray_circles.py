import cv2
import numpy as np

img_ans = cv2.imread("correct_answers/answer_01.jpg", cv2.IMREAD_GRAYSCALE)
img_puz = cv2.imread("puzzles/puzzle_extra_05.jpg", cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(img_ans, None)
kp_p, des_p = sift.detectAndCompute(img_puz, None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des_p, des_a)
good_matches = sorted(matches, key=lambda x: x.distance)[:100]
pts_p = np.float32([kp_p[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts_a = np.float32([kp_a[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H, _ = cv2.findHomography(pts_p, pts_a, cv2.RANSAC, 5.0)

# Warp puzzle to answer space
h_a, w_a = img_ans.shape[:2]
warped_puz = cv2.warpPerspective(img_puz, H, (w_a, h_a))

# Create mask for panel 1
h_p, w_p = img_puz.shape[:2]
panel_mask = np.zeros((h_p, w_p), dtype=np.uint8)
panel_mask[10:-10, 10:-10] = 255
warped_mask = cv2.warpPerspective(panel_mask, H, (w_a, h_a), flags=cv2.INTER_NEAREST)

# Added markings are gray (80-180) in answer, but the corresponding clean puzzle is white (>200)
mask_markings = (img_ans >= 60) & (img_ans <= 190) & (warped_puz > 200)
mask_markings = mask_markings.astype(np.uint8) * 255
mask_markings = cv2.bitwise_and(mask_markings, warped_mask)

# Morphological close/open to filter single-pixel noise
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_markings = cv2.morphologyEx(mask_markings, cv2.MORPH_CLOSE, k)
mask_markings = cv2.morphologyEx(mask_markings, cv2.MORPH_OPEN, k)

cnts, _ = cv2.findContours(mask_markings, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(cnts)} gray markings in answer_01.jpg:")
H_inv = np.linalg.inv(H)
for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    if area < 20:
        continue
    (cx, cy), r = cv2.minEnclosingCircle(c)
    pt = np.float32([cx, cy]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pt, H_inv).reshape(2)
    print(f"  GT {i+1}: center in answer=({cx:.1f}, {cy:.1f}), center in puzzle=({mapped[0]:.1f}, {mapped[1]:.1f}), r={r:.1f}, area={area:.1f}")
