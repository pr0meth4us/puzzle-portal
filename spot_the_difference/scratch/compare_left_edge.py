import cv2
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

puz = cv2.imread("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(puz)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
img_b_aligned, _, _ = std.align(img_a, img_b, skip_ecc=False)

# Crop left edge: x from 0 to 180, y from 400 to 600
left_a = img_a[400:600, 0:180]
left_b = img_b_aligned[400:600, 0:180]

# Concatenate side-by-side
left_vis = cv2.hconcat([left_a, left_b])
cv2.imwrite("/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/left_edge_puz7.png", left_vis)
print("Saved left edge crop to artifacts/left_edge_puz7.png")
