import cv2
import numpy as np
import spot_the_differences

# Load and slice puzzle_03
combined = spot_the_differences.load_bgr("validation_dataset/puzzle_03.jpg")
cropped, crop_y = spot_the_differences.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = spot_the_differences.auto_slice(cropped)

# Align
img_b_aligned, valid_y_range, H_align = spot_the_differences.align(img_a, img_b, skip_ecc=False)

# Detect
h, w = img_a.shape[:2]
thresh = cv2.absdiff(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_b_aligned, cv2.COLOR_BGR2GRAY))
# run color detection
split_dir = "vertical"
hsv = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
mean_sat = float(hsv[:, :, 1].mean())
is_colour = (mean_sat >= 20.0)
floor = 10.0

# Apply the same mask
bmask = np.zeros_like(thresh)
bmask[8:h - 8, 16:w - 8] = 255
if H_align is not None:
    mask_b = np.ones(img_b.shape[:2], dtype=np.uint8) * 255
    warped_mask_b = cv2.warpPerspective(mask_b, H_align, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    warped_mask_b = cv2.erode(warped_mask_b, kernel)
    bmask = cv2.bitwise_and(bmask, warped_mask_b)

# Otsu hue etc.
gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b_aligned, cv2.COLOR_BGR2GRAY)
diff_gray = cv2.absdiff(gray_a, gray_b)
_, otsu_t = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
otsu_t = cv2.bitwise_and(otsu_t, bmask)

# Delta map
cdiff_rgb = np.mean(cv2.absdiff(img_a, img_b).astype(np.float32), axis=2)

# Find contours
pre_cnts, _ = cv2.findContours(otsu_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(pre_cnts)} pre-contours.")
candidates = []
for i, c in enumerate(pre_cnts):
    (cx, cy), r = cv2.minEnclosingCircle(c)
    cx, cy, r = int(cx), int(cy), int(r)
    # compute delta
    mask = np.zeros_like(cdiff_rgb, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 1, cv2.FILLED)
    delta = float(cv2.mean(cdiff_rgb, mask=mask)[0])
    candidates.append((cx, cy, r, delta, i))

candidates = sorted(candidates, key=lambda x: x[3])
for cx, cy, r, delta, idx in candidates:
    print(f"Candidate {idx}: cx={cx}, cy={cy}, r={r}, delta={delta:.2f}")
