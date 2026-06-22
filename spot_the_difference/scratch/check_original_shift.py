import cv2
import numpy as np

img_ans = cv2.imread("correct_answers/answer_01.jpg")
img_puz = cv2.imread("puzzles/puzzle_extra_06.jpg")

sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(img_ans, None)
kp_p, des_p = sift.detectAndCompute(img_puz, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des_p, des_a)
good_matches = sorted(matches, key=lambda x: x.distance)[:100]

pts_p = np.float32([kp_p[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts_a = np.float32([kp_a[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(pts_p, pts_a, cv2.RANSAC, 5.0)
print("Homography matrix from puzzle_extra_06 to answer_01.jpg:")
print(H)

# Corners of puzzle_extra_06
h_p, w_p = img_puz.shape[:2]
corners = np.float32([[0, 0], [w_p, 0], [w_p, h_p], [0, h_p]]).reshape(-1, 1, 2)
mapped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
print("\nMapped corners in answer_01.jpg:")
for i, pt in enumerate(mapped_corners):
    print(f"  Corner {i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")
