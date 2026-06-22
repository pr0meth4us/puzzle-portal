import cv2
import numpy as np

img1 = cv2.imread("correct_answers/answer_01.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("puzzles/puzzle_extra_05.jpg", cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# Compute affine transform
M, inliers = cv2.estimateAffinePartial2D(pts2, pts1)
if M is not None:
    scale = np.sqrt(M[0,0]**2 + M[0,1]**2)
    angle = np.arctan2(M[0,1], M[0,0]) * 180 / np.pi
    tx, ty = M[0,2], M[1,2]
    print(f"Transformation from puzzle_extra_05 to answer_01:")
    print(f"  Scale: {scale:.4f}")
    print(f"  Rotation Angle: {angle:.2f} degrees")
    print(f"  Translation: ({tx:.2f}, {ty:.2f})")
else:
    print("Could not estimate transformation.")
