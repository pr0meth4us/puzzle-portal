import cv2
import sys

sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

combined = std.load_bgr("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
img_b_aligned, _, H = std.align(img_a, img_b, skip_ecc=False)

# Crop left region covering y=400 to y=600, x=0 to x=200
crop_a = img_a[400:600, 0:200]
crop_b = img_b_aligned[400:600, 0:200]
crop_b_orig = img_b[400:600, 0:200]

cv2.imwrite("scratch/crops/left_edge_a.png", crop_a)
cv2.imwrite("scratch/crops/left_edge_b_aligned.png", crop_b)
cv2.imwrite("scratch/crops/left_edge_b_orig.png", crop_b_orig)
cv2.imwrite("scratch/crops/left_edge_sidebyside.png", cv2.hconcat([crop_a, crop_b]))
cv2.imwrite("scratch/crops/left_edge_sidebyside_orig.png", cv2.hconcat([crop_a, crop_b_orig]))

print("Saved left edge comparison crops!")
