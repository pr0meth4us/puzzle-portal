import cv2
import os

img_a = cv2.imread("puzzles/puzzle_extra_05.jpg")
img_b = cv2.imread("puzzles/puzzle_extra_06.jpg")

coords = [
    (485, 25),
    (410, 39),
    (184, 53),
    (541, 59),
    (119, 73),
    (316, 85),
    (501, 88),
    (127, 129),
    (450, 147),
    (221, 225),
    (560, 279),
    (72, 319),
    (332, 368),
    (536, 402)
]

os.makedirs("scratch/crops", exist_ok=True)

for idx, (cx, cy) in enumerate(coords):
    r = 30
    y1, y2 = max(0, cy - r), min(img_a.shape[0], cy + r)
    x1, x2 = max(0, cx - r), min(img_a.shape[1], cx + r)
    
    crop_a = img_a[y1:y2, x1:x2]
    crop_b = img_b[y1:y2, x1:x2]
    
    # Concatenate side by side
    combined = cv2.hconcat([crop_a, crop_b])
    cv2.imwrite(f"scratch/crops/crop_{idx+1:02d}_{cx}_{cy}.png", combined)

print("Saved crops to scratch/crops/")
