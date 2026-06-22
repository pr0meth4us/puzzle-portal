import cv2
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

combined = std.load_bgr("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)

print(f"Original shape: {combined.shape}")
print(f"Cropped shape: {cropped_combined.shape}")
print(f"crop_y_offset: {crop_y_offset}")
print(f"img_a shape: {img_a.shape}")
print(f"img_b shape: {img_b.shape}")
print(f"a_start: {a_start}, b_start: {b_start}")
