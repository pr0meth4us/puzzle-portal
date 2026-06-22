import cv2
import numpy as np

img_puz = cv2.imread("puzzles/puzzle_extra_05.jpg")
img_ans = cv2.imread("correct_answers/answer_01.jpg")

if img_puz is None or img_ans is None:
    print("Could not read images.")
    exit()

print(f"Puzzle extra 5 shape: {img_puz.shape}")
print(f"Answer 1 shape: {img_ans.shape}")

# Resize img_puz to match img_ans size for direct pixel comparison after alignment
# Or we can align them using SIFT/Homography
sift = cv2.SIFT_create()
kp_p, des_p = sift.detectAndCompute(img_puz, None)
kp_a, des_a = sift.detectAndCompute(img_ans, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des_p, des_a)
pts_p = np.float32([kp_p[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts_a = np.float32([kp_a[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(pts_p, pts_a, cv2.RANSAC, 5.0)
h_a, w_a = img_ans.shape[:2]
warped_puz = cv2.warpPerspective(img_puz, H, (w_a, h_a))

# Compute absolute difference
diff = cv2.absdiff(img_ans, warped_puz)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Threshold to find markings
_, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

# Find contours of markings
cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(cnts)} difference regions between clean puzzle and answer key:")
marked_count = 0
for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    if area < 100:
        continue
    marked_count += 1
    (cx, cy), r = cv2.minEnclosingCircle(c)
    # Print the median color of this marking in img_ans
    mask_c = np.zeros(gray_diff.shape, dtype=np.uint8)
    cv2.drawContours(mask_c, [c], -1, 255, cv2.FILLED)
    mean_val = cv2.mean(img_ans, mask=mask_c)[:3]
    print(f"  Marking {marked_count}: center=({cx:.1f}, {cy:.1f}), r={r:.1f}, area={area:.1f}, mean BGR={mean_val}")
