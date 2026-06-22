import cv2
import numpy as np
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

puz = cv2.imread("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(puz)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)

# Let's crop a region in the sky/water or river where the watermark is visible
# For example: y in [100, 200], x in [100, 300]
sky_roi = img_a[100:200, 100:300]
hsv = cv2.cvtColor(sky_roi, cv2.COLOR_BGR2HSV)

# Save the sky ROI for visual reference
cv2.imwrite("/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/sky_roi.png", sky_roi)

# Print some stats about the HSV values
print("Sky ROI HSV stats:")
print(f"  H: mean={hsv[:,:,0].mean():.1f}, min={hsv[:,:,0].min()}, max={hsv[:,:,0].max()}")
print(f"  S: mean={hsv[:,:,1].mean():.1f}, min={hsv[:,:,1].min()}, max={hsv[:,:,1].max()}")
print(f"  V: mean={hsv[:,:,2].mean():.1f}, min={hsv[:,:,2].min()}, max={hsv[:,:,2].max()}")
