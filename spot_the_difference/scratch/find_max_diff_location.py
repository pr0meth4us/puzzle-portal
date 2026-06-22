import cv2
import numpy as np

crop_a = cv2.imread("/Users/nicksng/code/puzzle-portal/spot_the_difference/scratch/crop_river_a_correct.png")
crop_b = cv2.imread("/Users/nicksng/code/puzzle-portal/spot_the_difference/scratch/crop_river_b_aligned_correct.png")

diff = cv2.absdiff(crop_a, crop_b)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

max_val = np.max(gray_diff)
max_loc = np.unravel_index(np.argmax(gray_diff), gray_diff.shape)
print(f"Max grayscale diff value: {max_val} at local pixel (y={max_loc[0]}, x={max_loc[1]})")

# Global coordinate in img_a:
# cx, cy = 45, 421
# x1, x2 = max(0, cx - 50) = 0, min(w, cx + 50) = 95
# y1, y2 = max(0, cy - 50) = 371, min(h, cy + 50) = 471
global_x = max_loc[1] + 0
global_y = max_loc[0] + 371
print(f"Global coordinate in img_a: x={global_x}, y={global_y}")

# Print count of pixels with difference > 20, 30, 40
print(f"Number of pixels with diff > 20: {np.sum(gray_diff > 20)}")
print(f"Number of pixels with diff > 30: {np.sum(gray_diff > 30)}")
print(f"Number of pixels with diff > 40: {np.sum(gray_diff > 40)}")

# Let's list the top 10 local locations of highest differences
indices = np.argsort(gray_diff.ravel())[::-1][:10]
print("\nTop 10 differences in the crop:")
print("Rank | Local (y, x) | Global (y, x) | Diff Value")
print("-" * 50)
for rank, idx in enumerate(indices):
    y, x = np.unravel_index(idx, gray_diff.shape)
    val = gray_diff[y, x]
    print(f"{rank+1:4d} | ({y:3d}, {x:3d})   | ({y+371:3d}, {x:3d})   | {val}")
