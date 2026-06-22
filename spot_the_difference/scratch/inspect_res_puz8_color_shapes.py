import cv2
import numpy as np

img = cv2.imread("results/res_puzzle_08.png")
if img is None:
    print("Could not read results/res_puzzle_08.png")
    exit()

# Let's find the most common bright/saturated color in the image which represents our drawing color.
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Mask for highly saturated and bright colors
sat_mask = (s > 150) & (v > 150)
unique_hues, counts = np.unique(h[sat_mask], return_counts=True)
if len(unique_hues) == 0:
    print("No saturated drawing color found in res_puzzle_08.png")
    exit()

best_hue = unique_hues[np.argmax(counts)]
print(f"Detected drawing hue: {best_hue * 2} degrees")

# Create a mask for this exact hue
hue_mask = (h >= best_hue - 5) & (h <= best_hue + 5) & sat_mask
mask_img = np.zeros_like(h, dtype=np.uint8)
mask_img[hue_mask] = 255

cnts, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(cnts)} contours in drawing color:")
for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    if area < 30:
        continue
    peri = cv2.arcLength(c, True)
    circ = 4 * np.pi * area / (peri ** 2) if peri > 0 else 0
    (cx, cy), r = cv2.minEnclosingCircle(c)
    
    # Check if it is a circle or contour
    # A perfect circle has circularity > 0.85
    # Let's print circularity and bounding box
    print(f"  Shape {i+1}: Center=({cx:.1f}, {cy:.1f}), Radius={r:.1f}, Area={area:.1f}, Circularity={circ:.3f}")
