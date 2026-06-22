import cv2
import numpy as np

img = cv2.imread("spot_the_difference/correct_answers/answer_08.jpg")
if img is None:
    print("Could not read answer_08.jpg")
    exit()

# Let's find red pixels (since ground truth drawings are usually red)
# Red in BGR: B is low, G is low, R is high
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Red color range in HSV
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# Find contours
cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total red contours found in answer_08.jpg: {len(cnts)}")

for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    if area < 50:
        continue
    peri = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / (peri ** 2) if peri > 0 else 0
    x, y, w, h = cv2.boundingRect(c)
    print(f"  Contour {i+1}: area={area:.1f}, circularity={circularity:.3f}, bounding box=[{x}, {y}, {w}, {h}]")
