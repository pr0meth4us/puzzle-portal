import cv2
import numpy as np
from spot_the_differences import _find_separator

img = cv2.imread('validation_dataset/puzzle_03.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
prof_y = gray.mean(axis=1)
prof_x = gray.mean(axis=0)

h, w = gray.shape
print(f"h={h}, w={w}")
print("Y profile (cut horizontal):")
sep_y = _find_separator(prof_y, h)
print("X profile (cut vertical):")
sep_x = _find_separator(prof_x, w)
