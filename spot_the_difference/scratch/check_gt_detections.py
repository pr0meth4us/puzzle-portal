import cv2
import numpy as np
import sys
sys.path.append(".")
import engine_v3

img_a = engine_v3.load_bgr("puzzles/puzzle_extra_05.jpg")
img_b = engine_v3.load_bgr("puzzles/puzzle_extra_06.jpg")

img_b_aligned, H_align, valid_mask = engine_v3.align(img_a, img_b)

h, w = img_a.shape[:2]
gray_a = engine_v3._gray(img_a)
gray_b = engine_v3._gray(cv2.resize(img_b_aligned, (w, h), interpolation=cv2.INTER_LANCZOS4))
abs_diff = cv2.absdiff(gray_a, gray_b)

# Map ground truth coordinates from answer_01.jpg to puzzle_extra_05 space
# We computed them using the inverse homography:
gt_coords = {
    "GT 1 (Cloud)": (109, 64),
    "GT 2 (Cloud)": (545, 85),
    "GT 3 (Cloud)": (294, 87),
    "GT 4 (Cloud)": (439, 100),
    "GT 5 (Bird Beak)": (112, 127),
    "GT 6 (Baby Bird)": (207, 218),
    "GT 7 (Climber Belt)": (557, 290),
    "GT 8 (Spider)": (56, 312),
    "GT 9 (Elephant Ear/Tusk)": (322, 363),
    "GT 10 (Croc Mouth)": (517, 393)
}

print("Checking Ground Truth Regions:")
for name, (cx, cy) in gt_coords.items():
    # Find local max in absolute difference in a 30x30 window
    r = 20
    y1, y2 = max(0, cy - r), min(h, cy + r)
    x1, x2 = max(0, cx - r), min(w, cx + r)
    
    roi = abs_diff[y1:y2, x1:x2]
    max_val = np.max(roi)
    
    # Coordinates of max val in full image
    loc = np.unravel_index(np.argmax(roi), roi.shape)
    peak_y = y1 + loc[0]
    peak_x = x1 + loc[1]
    
    # Calculate SSIM local score
    print(f"  {name}: Center=({cx},{cy}) -> Local Peak=({peak_x},{peak_y}) with Diff={max_val}")
