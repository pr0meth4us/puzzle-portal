import cv2
import numpy as np
import sys

sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

combined = std.load_bgr("puzzles/puzzle_07.jpg")
cropped, y_off = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped)
img_b_aligned, valid_y, H = std.align(img_a, img_b, skip_ecc=False)

h, w = img_a.shape[:2]
diff = cv2.absdiff(img_a, img_b_aligned)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Apply a margin mask of x < 40 to completely ignore border noise
bmask = np.ones((h, w), dtype=np.uint8) * 255
bmask[:, :40] = 0
bmask[:, w - 40:] = 0
bmask[:32, :] = 0
bmask[h - 32:, :] = 0
bgr_sum = np.sum(img_b_aligned, axis=2)
bmask[bgr_sum < 15] = 0

gray_diff_masked = cv2.bitwise_and(gray_diff, bmask)

# Find all contours in the region x < 200, y between 400 and 600
sub_mask = np.zeros((h, w), dtype=np.uint8)
sub_mask[400:600, 40:200] = 255

thresh_map = cv2.bitwise_and(gray_diff_masked, sub_mask)
_, th = cv2.threshold(thresh_map, 10, 255, cv2.THRESH_BINARY)

# Morphological closing to merge blobs
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k_close)

cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(cnts)} difference blobs in the left area:")
for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    M = cv2.moments(c)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        pts = c.reshape(-1, 2)
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        print(f"  Blob {i+1}: Center=({cx:.1f}, {cy:.1f}), Area={area:.1f}, Bbox=({min_x}, {min_y}) -> ({max_x}, {max_y})")
