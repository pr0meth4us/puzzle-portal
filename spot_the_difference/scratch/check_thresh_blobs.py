import cv2
import numpy as np
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

puz = cv2.imread("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(puz)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
img_b_aligned, valid_y_range, H_align = std.align(img_a, img_b, skip_ecc=False)

h, w = img_a.shape[:2]
vy0, vy1 = valid_y_range if valid_y_range else (0, h)
vy0, vy1 = max(0, vy0), min(h, vy1)

gray_a = std._gray(img_a)
gray_b = std._gray(img_b_aligned)

score, diff = std.ssim(gray_a, gray_b, full=True)
inv = cv2.bitwise_not((diff * 255).clip(0, 255).astype(np.uint8))
lab_diff = std._lab_delta_map(img_a, img_b_aligned)
hue_diff_deg = std._hue_delta_map(img_a, img_b_aligned)

thresh_hue = (hue_diff_deg > std.HUE_FIXED_THRESH).astype(np.uint8) * 255
k_hue = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (std.HUE_DILATE_KSIZE, std.HUE_DILATE_KSIZE))
thresh_hue = cv2.morphologyEx(thresh_hue, cv2.MORPH_DILATE, k_hue)

valid_ssim = inv[vy0:vy1, :]
valid_lab  = lab_diff[vy0:vy1, :]
otsu_ssim  = cv2.threshold(valid_ssim, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
otsu_lab   = cv2.threshold(valid_lab,  0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]

_, thresh_ssim = cv2.threshold(inv,      otsu_ssim, 255, cv2.THRESH_BINARY)
_, thresh_lab  = cv2.threshold(lab_diff, otsu_lab,  255, cv2.THRESH_BINARY)

thresh = cv2.bitwise_or(thresh_ssim, thresh_lab)
thresh = cv2.bitwise_or(thresh,      thresh_hue)

bmask = np.ones((h, w), dtype=np.uint8) * 255
std._mask_margins(img_a, bmask)

# Save thresh before morphology
thresh_before = cv2.bitwise_and(thresh, bmask)
left_thresh_before = thresh_before[400:600, 0:180]
cv2.imwrite("/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/thresh_left_before.png", left_thresh_before)

# Apply morphology
k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thresh_after = cv2.morphologyEx(thresh_before, cv2.MORPH_CLOSE, k9)
thresh_after = cv2.morphologyEx(thresh_after, cv2.MORPH_OPEN,  k5)
thresh_after = cv2.bitwise_and(thresh_after, bmask)

left_thresh_after = thresh_after[400:600, 0:180]
cv2.imwrite("/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/thresh_left_after.png", left_thresh_after)

print("Saved left-edge thresh crops to artifacts directory.")
