import cv2
import numpy as np
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

def create_line_kernel(length, angle_deg):
    sz = length * 2 + 1
    kernel = np.zeros((sz, sz), dtype=np.uint8)
    angle_rad = np.deg2rad(angle_deg)
    dx = int(length * np.cos(angle_rad))
    dy = int(length * np.sin(angle_rad))
    cv2.line(kernel, (length - dx, length + dy), (length + dx, length - dy), 1, 1)
    return kernel

puz = cv2.imread("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(puz)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
img_b_aligned, valid_y_range, H_align = std.align(img_a, img_b, skip_ecc=False)

h, w = img_a.shape[:2]
vy0, vy1 = valid_y_range if valid_y_range else (0, h)

gray_a = std._gray(img_a)
gray_b = std._gray(img_b_aligned)
score, diff = std.ssim(gray_a, gray_b, full=True)
inv = cv2.bitwise_not((diff * 255).clip(0, 255).astype(np.uint8))
lab_diff = std._lab_delta_map(img_a, img_b_aligned)

otsu_ssim = cv2.threshold(inv[vy0:vy1, :], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
otsu_lab  = cv2.threshold(lab_diff[vy0:vy1, :], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]

# Standard thresholding
_, thresh_ssim = cv2.threshold(inv, otsu_ssim, 255, cv2.THRESH_BINARY)
_, thresh_lab  = cv2.threshold(lab_diff, otsu_lab, 255, cv2.THRESH_BINARY)
thresh = cv2.bitwise_or(thresh_ssim, thresh_lab)

# Mask margins
bmask = np.ones((h, w), dtype=np.uint8) * 255
std._mask_margins(img_a, bmask)
# Mask left margin slightly wider to avoid left SIFT alignment edge
bmask[:, :48] = 0

thresh = cv2.bitwise_and(thresh, bmask)

# Save threshold before line filtering
cv2.imwrite("/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/thresh_before_line.png", thresh)

# Create -30 degree line kernel (angle in OpenCV coordinate system is positive clockwise,
# so -30 degrees is 30 degrees counter-clockwise, i.e., 150 degrees)
kernel_line = create_line_kernel(15, -30)

# Morphological opening to detect lines
detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_line)

# Dilate the detected lines to cover the watermark completely
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
dilated_lines = cv2.dilate(detected_lines, kernel_dilate)

# Subtract dilated lines from thresh
thresh_filtered = cv2.bitwise_and(thresh, cv2.bitwise_not(dilated_lines))

cv2.imwrite("/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/thresh_after_line.png", thresh_filtered)
cv2.imwrite("/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/detected_lines.png", dilated_lines)

print("Line kernel shape:", kernel_line.shape)
print("Sum of thresh before:", thresh.sum() // 255)
print("Sum of detected lines:", dilated_lines.sum() // 255)
print("Sum of thresh after:", thresh_filtered.sum() // 255)
