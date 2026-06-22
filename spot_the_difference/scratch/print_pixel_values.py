import cv2
import numpy as np

img_a = cv2.imread("/Users/nicksng/code/puzzle-portal/spot_the_difference/scratch/crop_river_a_correct.png")
img_b = cv2.imread("/Users/nicksng/code/puzzle-portal/spot_the_difference/scratch/crop_river_b_aligned_correct.png")

# Let's print out BGR values of a row (y=50, which is global y=421) for x=0..60
y = 50
print("Local x | Global x | img_a BGR | img_b_aligned BGR | Abs Diff")
print("-" * 65)
for x in range(0, 60):
    val_a = img_a[y, x].tolist()
    val_b = img_b[y, x].tolist()
    d = [abs(val_a[i] - val_b[i]) for i in range(3)]
    print(f"x={x:2d}     | x={x:2d}     | {val_a} | {val_b} | {d} (mean: {np.mean(d):.1f})")
