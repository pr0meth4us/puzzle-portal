import cv2
import numpy as np

img_p5 = cv2.imread("puzzles/puzzle_extra_05.jpg")
img_p6 = cv2.imread("puzzles/puzzle_extra_06.jpg")
img_ans = cv2.imread("correct_answers/answer_01.jpg")

if any(img is None for img in (img_p5, img_p6, img_ans)):
    print("Could not read images.")
    exit()

h_a, w_a = img_ans.shape[:2]

# Align img_p5 to top half of img_ans
# Transformation: Scale = 0.7248, tx = 17.99, ty = 29.19
M5 = np.float32([[0.7248, 0, 17.99], [0, 0.7248, 29.19]])
warped_p5 = cv2.warpAffine(img_p5, M5, (w_a, h_a))

# Align img_p6 to bottom half of img_ans
# Transformation: Scale = 0.7249, tx = 25.91, ty = 406.49
M6 = np.float32([[0.7249, 0, 25.91], [0, 0.7249, 406.49]])
warped_p6 = cv2.warpAffine(img_p6, M6, (w_a, h_a))

# Combine warped panels to make a clean reconstructed image
clean_recon = np.zeros_like(img_ans)
# Top half mask (y < 380)
clean_recon[:380, :] = warped_p5[:380, :]
# Bottom half mask (y >= 380)
clean_recon[380:, :] = warped_p6[380:, :]

# Compute absolute difference between answer key and clean reconstructed image
diff = cv2.absdiff(img_ans, clean_recon)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Threshold to find markings (ignoring tiny shifts near edges by using a mask)
# Let's create a mask that excludes boundaries of the panels
mask_valid = np.zeros((h_a, w_a), dtype=np.uint8)
# Panel 5 valid region
x5_min, x5_max = int(18 + 5), int(18 + img_p5.shape[1] * 0.7248 - 5)
y5_min, y5_max = int(29 + 5), int(29 + img_p5.shape[0] * 0.7248 - 5)
mask_valid[y5_min:y5_max, x5_min:x5_max] = 255

# Panel 6 valid region
x6_min, x6_max = int(26 + 5), int(26 + img_p6.shape[1] * 0.7249 - 5)
y6_min, y6_max = int(406 + 5), int(406 + img_p6.shape[0] * 0.7249 - 5)
mask_valid[y6_min:y6_max, x6_min:x6_max] = 255

gray_diff = cv2.bitwise_and(gray_diff, mask_valid)
_, thresh = cv2.threshold(gray_diff, 45, 255, cv2.THRESH_BINARY)

# Morphological opening to remove noise
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)

cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(cnts)} real marking difference regions:")
count = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area < 10:
        continue
    count += 1
    (cx, cy), r = cv2.minEnclosingCircle(c)
    # Get BGR values in answer key
    mask_c = np.zeros(gray_diff.shape, dtype=np.uint8)
    cv2.drawContours(mask_c, [c], -1, 255, cv2.FILLED)
    mean_val = cv2.mean(img_ans, mask=mask_c)[:3]
    # round BGR
    mean_val = [round(v, 1) for v in mean_val]
    print(f"  Marking {count}: Center=({cx:.1f}, {cy:.1f}), Radius={r:.1f}, Area={area:.1f}, Mean BGR={mean_val}")
