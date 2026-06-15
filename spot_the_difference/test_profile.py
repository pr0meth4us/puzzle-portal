import cv2
import numpy as np

img = cv2.imread('validation_dataset/puzzle_03.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
prof_y = gray.mean(axis=1)
prof_x = gray.mean(axis=0)

print(f"Top 20 rows brightness (Y): {prof_y[:20].round()}")
print(f"Bottom 20 rows brightness (Y): {prof_y[-20:].round()}")
print(f"Left 20 cols brightness (X): {prof_x[:20].round()}")
print(f"Right 20 cols brightness (X): {prof_x[-20:].round()}")

