import cv2
import numpy as np

img_puz = cv2.imread("validation_dataset/puzzle_05.jpg")
if img_puz is None:
    print("Could not read puzzle_05.jpg")
    exit()

# Slice
h, w = img_puz.shape[:2]
panel_h = 555
panel_a = img_puz[1:1+panel_h, :]
panel_b = img_puz[556:556+panel_h, :]

# ROI: x from 500 to 565, y from 480 to 555
roi_a = panel_a[480:555, 500:565]
roi_b = panel_b[480:555, 500:565]

diff = cv2.absdiff(roi_a, roi_b)
print(f"Mean ROI difference: {np.mean(diff)}")

# Let's save the crops so we can inspect their content or write a script to check if there is a fish shape
# Wait, let's check the number of non-zero pixels
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)
print(f"Number of difference pixels (thresh > 15): {np.sum(thresh == 255)}")
