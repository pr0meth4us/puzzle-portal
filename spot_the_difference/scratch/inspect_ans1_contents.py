import cv2
import os

img_ans = cv2.imread("correct_answers/answer_01.jpg", cv2.IMREAD_GRAYSCALE)
if img_ans is None:
    print("Could not read correct_answers/answer_01.jpg")
    exit()

sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(img_ans, None)

best_matches = 0
best_file = None

# Check validation_dataset
for folder in ["validation_dataset", "puzzles"]:
    for f in os.listdir(folder):
        if not f.endswith((".png", ".jpg")):
            continue
        p = os.path.join(folder, f)
        img_puz = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img_puz is None:
            continue
        kp_p, des_p = sift.detectAndCompute(img_puz, None)
        if des_p is None:
            continue
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des_a, des_p)
        good_matches = [m for m in matches if m.distance < 150]
        print(f"Matches for {f}: {len(good_matches)}")
        if len(good_matches) > best_matches:
            best_matches = len(good_matches)
            best_file = f

print(f"\nBest match for answer_01.jpg is: {best_file} with {best_matches} SIFT matches")
