import cv2
import numpy as np

puz = cv2.imread("validation_dataset/puzzle_03.jpg")
ans = cv2.imread("correct_answers/answer_03.jpg")

# Region of interest around Diff 11-13: y in [130, 150], x in [95, 170]
roi_puz = puz[130:155, 95:175]
roi_ans = ans[130:155, 95:175]

print("Puzzle ROI mean color:", np.mean(roi_puz, axis=(0,1)))
print("Answer ROI mean color:", np.mean(roi_ans, axis=(0,1)))

# Let's count where the difference is large (>30)
diff = cv2.absdiff(roi_puz, roi_ans)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
y_coords, x_coords = np.where(gray_diff > 30)

print(f"Number of pixels with difference > 30: {len(y_coords)}")
if len(y_coords) > 0:
    # Let's see some sample pixels in both
    for i in range(min(5, len(y_coords))):
        py, px = y_coords[i], x_coords[i]
        ay, ax = py + 130, px + 95
        print(f"Pixel at ({ay}, {ax}): Puzzle BGR={puz[ay, ax].tolist()}, Answer BGR={ans[ay, ax].tolist()}")
