import cv2
import numpy as np

crop_a = cv2.imread("/Users/nicksng/code/puzzle-portal/spot_the_difference/scratch/crop_river_a_correct.png")
crop_b = cv2.imread("/Users/nicksng/code/puzzle-portal/spot_the_difference/scratch/crop_river_b_aligned_correct.png")

diff = cv2.absdiff(crop_a, crop_b)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Downsample by 5x5 to print a readable text map of differences
h, w = gray_diff.shape
sh, sw = h // 5, w // 5
downsampled = np.zeros((sh, sw), dtype=np.uint8)

for r in range(sh):
    for c in range(sw):
        downsampled[r, c] = np.mean(gray_diff[r*5:(r+1)*5, c*5:(c+1)*5])

print("Difference map (each cell is 5x5 pixels, value shows mean absolute difference):")
print("   " + "".join(f"{c*5:4d}" for c in range(sw)))
for r in range(sh):
    row_str = f"{r*5:2d} |"
    for c in range(sw):
        val = downsampled[r, c]
        if val < 5:
            row_str += "    "
        elif val < 15:
            row_str += " .  "
        elif val < 30:
            row_str += " *  "
        elif val < 60:
            row_str += " #  "
        else:
            row_str += " @  "
    print(row_str)
