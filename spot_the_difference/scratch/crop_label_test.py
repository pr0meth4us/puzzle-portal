import cv2

puz = cv2.imread("puzzles/puzzle_07.jpg")
cropped, _ = cv2.crop_text_by_gap(puz) if hasattr(cv2, "crop_text_by_gap") else (puz, 0)
# Let's import spot_the_differences to slice correctly
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

cropped_combined, crop_y_offset = std.crop_text_by_gap(puz)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)

# Crop 100x100 in img_a around (1036, 430) and (1095, 449)
for idx, (cx, cy) in enumerate([(1036, 430), (1095, 449)]):
    y1, y2 = max(0, cy - 50), min(img_a.shape[0], cy + 50)
    x1, x2 = max(0, cx - 50), min(img_a.shape[1], cx + 50)
    crop = img_a[y1:y2, x1:x2]
    cv2.imwrite(f"/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/label_crop_{idx}.png", crop)
    print(f"Saved crop {idx} to artifacts/label_crop_{idx}.png")
