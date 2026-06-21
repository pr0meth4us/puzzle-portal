import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
img = cv2.imread(str(SCRIPT_DIR / "validation_dataset/puzzle_06.jpg"))
h, w = img.shape[:2]
sep = h // 2
img_a = img[:sep]

# Crop using the same logic as spot_the_differences
gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 8)
if np.mean(binary) < 128:
    binary = cv2.bitwise_not(binary)

cnts, _ = cv2.findContours(cv2.bitwise_not(binary),
                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(cnts, key=cv2.contourArea)
x, y, cw, ch = cv2.boundingRect(largest)
print(f"Bounding box: x={x}, y={y}, w={cw}, h={ch}")

# Crop the grid region
grid_crop = binary[y:y+ch, x:x+cw]

# Let's compute horizontal and vertical projections
# Sum vertically (along columns) and horizontally (along rows)
# Since the grid lines are black (intensity 0 in binary image, or we can use the bitwise_not of binary which has lines as 255)
# In cv2.bitwise_not(binary), lines and text are 255, background is 0.
grid_inv = cv2.bitwise_not(binary)[y:y+ch, x:x+cw]

row_sums = np.sum(grid_inv, axis=1)
col_sums = np.sum(grid_inv, axis=0)

# Let's print out some statistics to find local minima/maxima or peaks
# Since the grid lines run all the way across, horizontal and vertical grid lines will correspond to huge peaks in row_sums and col_sums!
# Let's find peaks in row_sums and col_sums that are separated by a minimum distance.
# If scipy is not available, we can write a simple peak finder.

def find_peaks_simple(arr, min_dist, threshold):
    peaks = []
    for i in range(1, len(arr)-1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
            if arr[i] > threshold:
                # check min distance
                if not peaks or (i - peaks[-1]) >= min_dist:
                    peaks.append(i)
                elif arr[i] > arr[peaks[-1]]:
                    peaks[-1] = i
    return peaks

# Grid line peak finder
# Grid lines are thin and continuous, so their sums will be very high.
row_peaks = find_peaks_simple(row_sums, min_dist=20, threshold=row_sums.max() * 0.5)
col_peaks = find_peaks_simple(col_sums, min_dist=20, threshold=col_sums.max() * 0.5)

print(f"Detected horizontal line positions (row peaks): {row_peaks} (count: {len(row_peaks)})")
print(f"Detected vertical line positions (col peaks): {col_peaks} (count: {len(col_peaks)})")

# The number of rows in the grid is len(row_peaks) - 1 if the lines form the boundaries
# Let's print the spacings
print("Row spacings:", np.diff(row_peaks))
print("Col spacings:", np.diff(col_peaks))
