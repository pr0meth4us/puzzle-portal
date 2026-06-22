import cv2
import numpy as np

img = cv2.imread("correct_answers/answer_01.jpg")
if img is None:
    print("Could not read correct_answers/answer_01.jpg")
    exit()

# Print unique colors that are not grey/black/white
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
# Filter for saturated colors (S > 50, V > 50)
sat_mask = (s > 50) & (v > 50)
unique_hues, counts = np.unique(h[sat_mask], return_counts=True)
print("Saturated hues and counts:")
for hue, count in sorted(zip(unique_hues, counts), key=lambda x: -x[1])[:10]:
    print(f"  Hue {hue * 2} deg: {count} pixels")
