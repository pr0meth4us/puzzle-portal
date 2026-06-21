import cv2
import numpy as np
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent
answers_dir = SCRIPT_DIR / "correct_answers"
val_dir = SCRIPT_DIR / "validation_dataset"
puzzles_dir = SCRIPT_DIR / "puzzles"

def get_image_files(directory):
    return [p for p in directory.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]

answer_files = get_image_files(answers_dir)
puzzle_files = get_image_files(val_dir) + get_image_files(puzzles_dir)

# Remove duplicate paths (since some files exist in both validation_dataset and puzzles)
seen = set()
unique_puzzle_files = []
for p in puzzle_files:
    if p.name not in seen and not p.name.startswith("answer"):
        seen.add(p.name)
        unique_puzzle_files.append(p)

print(f"Loaded {len(answer_files)} answers and {len(unique_puzzle_files)} unique puzzles.")

sift = cv2.SIFT_create()

for ans_path in sorted(answer_files):
    ans_img = cv2.imread(str(ans_path))
    if ans_img is None:
        continue
    gray_ans = cv2.cvtColor(ans_img, cv2.COLOR_BGR2GRAY)
    kp_ans, des_ans = sift.detectAndCompute(gray_ans, None)
    
    best_match_name = None
    best_inliers = 0
    
    for puz_path in unique_puzzle_files:
        puz_img = cv2.imread(str(puz_path))
        if puz_img is None:
            continue
        # Since puzzle might be combined or separate, let's just match the whole image
        gray_puz = cv2.cvtColor(puz_img, cv2.COLOR_BGR2GRAY)
        kp_puz, des_puz = sift.detectAndCompute(gray_puz, None)
        
        if des_ans is None or des_puz is None:
            continue
            
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_ans, des_puz, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
                
        if len(good) > 10:
            src_pts = np.float32([kp_ans[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_puz[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(np.sum(mask))
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_match_name = puz_path.name
                    
    print(f"Answer: {ans_path.name} -> Best Match: {best_match_name} (inliers: {best_inliers})")
