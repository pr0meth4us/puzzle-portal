import cv2
import numpy as np

ans = cv2.imread("correct_answers/answer_07.jpg")
h, w = ans.shape[:2]

# Let's filter for red strictly
hsv = cv2.cvtColor(ans, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
red_mask = mask1 | mask2

cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
dots = []
for c in cnts:
    area = cv2.contourArea(c)
    if area > 10:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if 4 < r < 40:
            dots.append((cx, cy, r))

# Deduplicate
filtered_dots = []
for cx, cy, r in dots:
    too_close = False
    for kx, ky, _ in filtered_dots:
        if np.hypot(cx - kx, cy - ky) < 15:
            too_close = True
            break
    if not too_close:
        filtered_dots.append((cx, cy, r))

print(f"Deduplicated red dots: {len(filtered_dots)}")
crops = []
for idx, (cx, cy, r) in enumerate(filtered_dots):
    cx, cy = int(cx), int(cy)
    y1, y2 = max(0, cy - 30), min(h, cy + 30)
    x1, x2 = max(0, cx - 30), min(w, cx + 30)
    crop = ans[y1:y2, x1:x2].copy()
    if crop.shape != (60, 60, 3):
        crop = cv2.resize(crop, (60, 60))
    cv2.putText(crop, str(idx+1), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    crops.append(crop)

# Save as a grid
grid_cols = 6
grid_rows = (len(crops) + grid_cols - 1) // grid_cols
grid_list = []
for r in range(grid_rows):
    row_crops = crops[r*grid_cols:(r+1)*grid_cols]
    while len(row_crops) < grid_cols:
        row_crops.append(np.zeros((60, 60, 3), dtype=np.uint8))
    grid_list.append(cv2.hconcat(row_crops))

if grid_list:
    grid = cv2.vconcat(grid_list)
    cv2.imwrite("/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/ans7_red_blobs.png", grid)
    print("Saved to artifacts/ans7_red_blobs.png")
