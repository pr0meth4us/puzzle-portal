import cv2
import numpy as np

img_puz = cv2.imread("validation_dataset/puzzle_05.jpg")
img_ans = cv2.imread("correct_answers/answer_05.jpg")

if img_puz is None or img_ans is None:
    print("Could not read images.")
    exit()

sift = cv2.SIFT_create()
kp_p, des_p = sift.detectAndCompute(img_puz, None)
kp_a, des_a = sift.detectAndCompute(img_ans, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des_p, des_a)
pts_p = np.float32([kp_p[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts_a = np.float32([kp_a[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
H_p_to_a, _ = cv2.findHomography(pts_p, pts_a, cv2.RANSAC, 5.0)

# Let's map both (543, 515) in Panel A and (543, 515) in Panel B to answer_05
pt_a = np.float32([543, 515]).reshape(-1, 1, 2)
pt_b = np.float32([543, 555 + 515]).reshape(-1, 1, 2)

map_a = cv2.perspectiveTransform(pt_a, H_p_to_a).reshape(2)
map_b = cv2.perspectiveTransform(pt_b, H_p_to_a).reshape(2)

print(f"Mapped Panel A fish coordinates: ({map_a[0]:.1f}, {map_a[1]:.1f})")
print(f"Mapped Panel B fish coordinates: ({map_b[0]:.1f}, {map_b[1]:.1f})")

# Let's list the 8 red circles in answer_05.jpg and check distances
hsv = cv2.cvtColor(img_ans, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
mask = mask1 | mask2
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("\nGround Truth Circles in answer_05.jpg:")
for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    if area < 50:
        continue
    (cx, cy), r = cv2.minEnclosingCircle(c)
    dist_a = np.sqrt((cx - map_a[0])**2 + (cy - map_a[1])**2)
    dist_b = np.sqrt((cx - map_b[0])**2 + (cy - map_b[1])**2)
    print(f"  GT {i+1}: Center=({cx:.1f}, {cy:.1f}), r={r:.1f}, Dist to Panel A Fish={dist_a:.1f}, Dist to Panel B Fish={dist_b:.1f}")
