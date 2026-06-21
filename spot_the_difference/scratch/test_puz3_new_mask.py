import cv2
import numpy as np
import spot_the_differences

combined = spot_the_differences.load_bgr("validation_dataset/puzzle_03.jpg")
cropped, crop_y = spot_the_differences.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = spot_the_differences.auto_slice(cropped)
img_b_aligned, valid_y_range, H_align = spot_the_differences.align(img_a, img_b)

# Run detection with custom logic
h, w = img_a.shape[:2]
thresh = cv2.absdiff(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_b_aligned, cv2.COLOR_BGR2GRAY))

# Otsu hue etc.
gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b_aligned, cv2.COLOR_BGR2GRAY)
diff_gray = cv2.absdiff(gray_a, gray_b)
_, otsu_t = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Apply proposed mask
bmask = np.zeros_like(thresh)
by_top, by_bot = 12, 20
bmask[by_top:h - by_bot, 16:w - 10] = 255

if H_align is not None:
    mask_b = np.ones(img_b.shape[:2], dtype=np.uint8) * 255
    warped_mask_b = cv2.warpPerspective(mask_b, H_align, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    warped_mask_b = cv2.erode(warped_mask_b, kernel)
    bmask = cv2.bitwise_and(bmask, warped_mask_b)

otsu_t = cv2.bitwise_and(otsu_t, bmask)

# Detect with min_area=30
circles, count = spot_the_differences.detect(
    img_a, img_b_aligned,
    min_area=30,
    delta_floor=7.0,
    valid_y_range=valid_y_range,
    split_dir="vertical",
    H=H_align
)

# Wait, detect function internally computes its own mask!
# So we need to make sure detect function in spot_the_differences.py is updated with this mask first.
