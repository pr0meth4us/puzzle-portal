import cv2
import numpy as np
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

fps = [
    (838.5, 659.8, "FP_838_660"),
    (904.0, 885.5, "FP_904_885"),
    (1043.0, 619.0, "FP_1043_619"),
    (171.0, 302.0, "FP_171_302"),
    (148.0, 393.0, "FP_148_393"),
    (38.0, 273.0, "FP_38_273")
]

puz = cv2.imread("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(puz)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
img_b_aligned, _, _ = std.align(img_a, img_b, skip_ecc=False)

rows = []
for i, (tx, ty, name) in enumerate(fps):
    tx, ty = int(tx), int(ty)
    y1, y2 = max(0, ty - 50), min(img_a.shape[0], ty + 50)
    x1, x2 = max(0, tx - 50), min(img_a.shape[1], tx + 50)
    
    crop_a = img_a[y1:y2, x1:x2].copy()
    crop_b = img_b_aligned[y1:y2, x1:x2].copy()
    
    if crop_a.shape != (100, 100, 3):
        crop_a = cv2.resize(crop_a, (100, 100))
        crop_b = cv2.resize(crop_b, (100, 100))
        
    cv2.putText(crop_a, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    pair = cv2.hconcat([crop_a, crop_b])
    rows.append(pair)

grid = cv2.vconcat(rows)
cv2.imwrite("/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/all_fps.png", grid)
print("Saved FP comparison grid to artifacts/all_fps.png")
