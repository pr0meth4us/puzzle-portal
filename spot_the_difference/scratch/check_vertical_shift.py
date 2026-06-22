import cv2
import numpy as np

# Load the images
puz = cv2.imread("/Users/nicksng/code/puzzle-portal/spot_the_difference/puzzles/puzzle_07.jpg")
ans = cv2.imread("/Users/nicksng/code/puzzle-portal/spot_the_difference/correct_answers/answer_07.jpg")

print(f"Puzzle shape: {puz.shape}")
print(f"Answer shape: {ans.shape}")

# Resize answer to match puzzle's width (1152) and see what the height becomes
scale = 1152 / ans.shape[1]
h_resized = int(ans.shape[0] * scale)
ans_resized = cv2.resize(ans, (1152, h_resized))
print(f"Resized answer shape: {ans_resized.shape}")

# Let's save a side-by-side comparison of the top-left area to check alignment
# We'll save a 400x400 crop from both
cv2.imwrite("/Users/nicksng/code/puzzle-portal/spot_the_difference/scratch/check_align_puz.png", puz[0:400, 0:400])
cv2.imwrite("/Users/nicksng/code/puzzle-portal/spot_the_difference/scratch/check_align_ans.png", ans_resized[0:400, 0:400])

# Let's find some prominent feature (like a corner or rock) in both and print its coordinate
# We can do template matching or just visual inspection.
# Let's also print the BGR values along a vertical line at x=200 for y=0..300 in both
print("\ny-coord | puzzle BGR | resized answer BGR")
print("-" * 45)
for y in range(0, 300, 20):
    val_p = puz[y, 200].tolist()
    val_a = ans_resized[y, 200].tolist()
    print(f"y={y:3d}   | {val_p} | {val_a}")
