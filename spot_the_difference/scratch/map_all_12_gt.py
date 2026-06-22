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

H, _ = cv2.findHomography(pts_p, pts_a, cv2.RANSAC, 5.0)
H_inv = np.linalg.inv(H)

gt_in_ans = [
    (105, 453),
    (421, 468),
    (239, 470),
    (344, 479),
    (107, 499),
    (176, 565),
    (430, 618),
    (66, 634),
    (259, 671),
    (401, 693)
]

print("Ground Truth in puzzle_extra_06 space:")
gt_mapped = []
for i, (cx, cy) in enumerate(gt_in_ans):
    pt = np.float32([cx, cy]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pt, H_inv).reshape(2)
    print(f"  GT {i+1}: ({mapped[0]:.1f}, {mapped[1]:.1f})")
    gt_mapped.append((mapped[0], mapped[1]))

our_detections = [
    (406, 53),
    (447, 145),
    (489, 32),
    (127, 129),
    (221, 225),
    (493, 88),
    (72, 319),
    (119, 73),
    (316, 85),
    (369, 95)
]

print("\nMatching our detections to GT (dist threshold = 50px):")
for idx, (ocx, ocy) in enumerate(our_detections):
    closest_dist = float('inf')
    closest_gt = -1
    for i, (gcx, gcy) in enumerate(gt_mapped):
        dist = np.sqrt((ocx - gcx)**2 + (ocy - gcy)**2)
        if dist < closest_dist:
            closest_dist = dist
            closest_gt = i
    status = "FALSE POSITIVE"
    if closest_dist < 50:
        status = f"MATCHED GT {closest_gt+1} (dist={closest_dist:.1f})"
    print(f"  Detection {idx+1} at ({ocx}, {ocy}) -> {status}")
