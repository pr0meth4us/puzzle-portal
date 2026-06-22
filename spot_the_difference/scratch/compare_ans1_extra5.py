import cv2
import numpy as np

img1 = cv2.imread("correct_answers/answer_01.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("puzzles/puzzle_extra_05.jpg", cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Could not read images.")
    exit()

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

good_matches = [m for m in matches if m.distance < 150]
print(f"Number of good SIFT matches between answer_01.jpg and puzzle_extra_05.jpg: {len(good_matches)}")
