import cv2
import numpy as np

img_puz = cv2.imread("validation_dataset/puzzle_05.jpg")
img_ans = cv2.imread("correct_answers/answer_05.jpg")

if img_puz is None or img_ans is None:
    print("Could not read images.")
    exit()

# SIFT matching to find homography
sift = cv2.SIFT_create()
kp_p, des_p = sift.detectAndCompute(img_puz, None)
kp_a, des_a = sift.detectAndCompute(img_ans, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des_p, des_a)
pts_p = np.float32([kp_p[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts_a = np.float32([kp_a[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
H_p_to_a, _ = cv2.findHomography(pts_p, pts_a, cv2.RANSAC, 5.0)

# Map (543, 516) to answer_05 space
pt = np.float32([543, 516]).reshape(-1, 1, 2)
mapped = cv2.perspectiveTransform(pt, H_p_to_a).reshape(2)
ax, ay = int(mapped[0]), int(mapped[1])
print(f"Mapped coordinates in answer_05: ({ax}, {ay})")

# Let's crop a 100x100 region around (ax, ay) in answer_05.jpg and check for red pixels
roi = img_ans[max(0, ay-50):min(img_ans.shape[0], ay+50), max(0, ax-50):min(img_ans.shape[1], ax+50)]
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
mask_red = mask1 | mask2

print(f"Red pixels in 100x100 ROI around fish: {np.sum(mask_red == 255)}")
