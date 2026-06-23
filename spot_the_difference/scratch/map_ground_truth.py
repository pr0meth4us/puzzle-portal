import cv2
import numpy as np

# Load images
p5 = cv2.imread("/Users/nicksng/code/puzzle-portal/spot_the_difference/puzzles/puzzle_extra_05.jpg")
ans = cv2.imread("/Users/nicksng/code/puzzle-portal/spot_the_difference/correct_answers/answer_01.jpg")

h_ans, w_ans = ans.shape[:2]
h_p5, w_p5 = p5.shape[:2]

# Bottom panel of answer_01
ph = h_ans // 2
panel_b = ans[ph:, :]

print(f"p5 shape: {p5.shape}, panel_b shape: {panel_b.shape}")

# SIFT matching
sift = cv2.SIFT_create()
kp_ref, des_ref = sift.detectAndCompute(p5, None)
kp_tgt, des_tgt = sift.detectAndCompute(panel_b, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des_ref, des_tgt, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

print(f"SIFT good matches: {len(good)}")

pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
pts_tgt = np.float32([kp_tgt[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

H, inliers = cv2.findHomography(pts_tgt, pts_ref, cv2.RANSAC, 5.0)
print(f"Homography inliers: {np.sum(inliers)}")

# Red circles detected in answer_01 space (bottom panel y = cy - ph)
red_circles_ans = [
    (105, 453 - ph),
    (421, 468 - ph),
    (239, 470 - ph),
    (344, 479 - ph),
    (107, 499 - ph),
    (176, 565 - ph),
    (430, 618 - ph),
    (66, 634 - ph),
    (259, 671 - ph),
    (401, 693 - ph)
]

# Project to p5 space
pts_tgt_circles = np.float32([[cx, cy] for cx, cy in red_circles_ans]).reshape(-1, 1, 2)
pts_projected = cv2.perspectiveTransform(pts_tgt_circles, H).reshape(-1, 2)

print("\nGround Truth mapped to puzzle_extra_05 space:")
mapped_gt = []
for idx, (mx, my) in enumerate(pts_projected):
    print(f"  GT {idx+1}: cx={int(mx)}, cy={int(my)}")
    mapped_gt.append((int(mx), int(my)))
