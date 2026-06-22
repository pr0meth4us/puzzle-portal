import cv2
import numpy as np

img_puz = cv2.imread("puzzles/puzzle_extra_05.jpg")
img_ans = cv2.imread("correct_answers/answer_01.jpg")

if img_puz is None or img_ans is None:
    print("Could not read images.")
    exit()

h_a, w_a = img_ans.shape[:2]
# Split answer image into top and bottom halves
top_half = img_ans[:h_a//2, :]
bot_half = img_ans[h_a//2:, :]

# Let's compare top_half to img_puz
sift = cv2.SIFT_create()
kp_p, des_p = sift.detectAndCompute(img_puz, None)
kp_t, des_t = sift.detectAndCompute(top_half, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des_p, des_t)
pts_p = np.float32([kp_p[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts_t = np.float32([kp_t[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(pts_p, pts_t, cv2.RANSAC, 5.0)
h_t, w_t = top_half.shape[:2]
warped_puz = cv2.warpPerspective(img_puz, H, (w_t, h_t))

# Compute difference
diff = cv2.absdiff(top_half, warped_puz)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)

cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Top half vs extra_05: Found {len(cnts)} difference regions:")
count = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area < 15:
        continue
    count += 1
    (cx, cy), r = cv2.minEnclosingCircle(c)
    # Print mean BGR of this region in top_half
    mask_c = np.zeros(gray_diff.shape, dtype=np.uint8)
    cv2.drawContours(mask_c, [c], -1, 255, cv2.FILLED)
    mean_val = cv2.mean(top_half, mask=mask_c)[:3]
    print(f"  Region {count}: center=({cx:.1f}, {cy:.1f}), r={r:.1f}, area={area:.1f}, BGR={mean_val}")
