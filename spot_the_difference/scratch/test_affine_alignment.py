import cv2
import numpy as np
import sys

sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

puz = cv2.imread("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(puz)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)

h, w = img_a.shape[:2]
g_a = std._gray(img_a)
g_b = std._gray(img_b)

# SIFT matching
sift = cv2.SIFT_create(nfeatures=5000)
kp_a, des_a = sift.detectAndCompute(g_a, None)
kp_b, des_b = sift.detectAndCompute(g_b, None)

matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = matcher.knnMatch(des_b, des_a, k=2)
good = [m for m, n in matches if m.distance < 0.75 * n.distance]

src = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

# 1. Homography
H_homo, mask_homo = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
img_b_homo = cv2.warpPerspective(img_b, H_homo, (w, h))
ssim_homo = std._ssim(img_a, img_b_homo)
print(f"SIFT + Homography: SSIM = {ssim_homo:.4f}")

# 2. Affine Partial 2D (Translation + Rotation + Uniform Scale)
M_affine, mask_affine = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
img_b_affine = cv2.warpAffine(img_b, M_affine, (w, h))
ssim_affine = std._ssim(img_a, img_b_affine)
print(f"SIFT + Affine Partial 2D: SSIM = {ssim_affine:.4f}")

# 3. Translation-only (Median shift)
shifts_x = [d[0][0] - s[0][0] for s, d in zip(src, dst)]
shifts_y = [d[0][1] - s[0][1] for s, d in zip(src, dst)]
tx = np.median(shifts_x)
ty = np.median(shifts_y)
M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
img_b_trans = cv2.warpAffine(img_b, M_trans, (w, h))
ssim_trans = std._ssim(img_a, img_b_trans)
print(f"SIFT + Translation-only (tx={tx:.2f}, ty={ty:.2f}): SSIM = {ssim_trans:.4f}")
