import cv2
import numpy as np

img = cv2.imread('validation_dataset/puzzle_03.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find bounding box of non-black and non-white pixels
# Actually, let's just use edge detection to find the main content
edges = cv2.Canny(gray, 50, 150)
coords = cv2.findNonZero(edges)
x, y, w, h = cv2.boundingRect(coords)

cropped = img[y:y+h, x:x+w]
print(f"Original: {img.shape}, Cropped: {cropped.shape}")
cv2.imwrite('cropped.jpg', cropped)
