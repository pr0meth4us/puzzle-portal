import cv2
import numpy as np

img_ans = cv2.imread("correct_answers/answer_01.jpg")
img_puz = cv2.imread("puzzles/puzzle_extra_05.jpg")

# SIFT match to find homography
sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(img_ans, None)
kp_p, des_p = sift.detectAndCompute(img_puz, None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des_p, des_a)
good_matches = sorted(matches, key=lambda x: x.distance)[:100]
pts_p = np.float32([kp_p[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts_a = np.float32([kp_a[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H, _ = cv2.findHomography(pts_p, pts_a, cv2.RANSAC, 5.0)

# Warp puzzle_extra_05 to answer_01 space
h_a, w_a = img_ans.shape[:2]
warped_puz = cv2.warpPerspective(img_puz, H, (w_a, h_a))

# Create a mask for the panel area in answer_01
h_p, w_p = img_puz.shape[:2]
panel_mask = np.zeros((h_p, w_p), dtype=np.uint8)
panel_mask[5:-5, 5:-5] = 255 # exclude borders
warped_mask = cv2.warpPerspective(panel_mask, H, (w_a, h_a), flags=cv2.INTER_NEAREST)

# Direct comparison inside the panel area
diff = cv2.absdiff(img_ans, warped_puz)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
gray_diff = cv2.bitwise_and(gray_diff, warped_mask)

# Find areas of significant difference
_, thresh = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Difference regions between answer_01 panel 1 and clean puzzle_extra_05:")
count = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area < 10:
         continue
    count += 1
    (cx, cy), r = cv2.minEnclosingCircle(c)
    # Find average BGR of this difference region in answer_01
    mask_c = np.zeros(gray_diff.shape, dtype=np.uint8)
    cv2.drawContours(mask_c, [c], -1, 255, cv2.FILLED)
    mean_ans = cv2.mean(img_ans, mask=mask_c)[:3]
    mean_puz = cv2.mean(warped_puz, mask=mask_c)[:3]
    print(f"  Marking {count}: center=({cx:.1f}, {cy:.1f}), r={r:.1f}, area={area:.1f}")
    print(f"    Answer BGR: {mean_ans}")
    print(f"    Puzzle BGR: {mean_puz}")
