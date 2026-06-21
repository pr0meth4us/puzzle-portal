import cv2
import spot_the_differences
from pathlib import Path

val_dir = Path("validation_dataset")
out_dir = Path("debug_out")
out_dir.mkdir(exist_ok=True)

# Puzzle 4
img4 = spot_the_differences.load_bgr(str(val_dir / "puzzle_04.jpg"))
img_a, img_b, _, _ = spot_the_differences.auto_slice(img4)
img_b_aligned, _, _ = spot_the_differences.align(img_a, img_b, skip_ecc=False)
circles4, _ = spot_the_differences.detect_line(img_a, img_b_aligned)

out4 = img_a.copy()
for cx, cy, r in circles4:
    cv2.circle(out4, (cx, cy), r, (0, 0, 255), 3)
cv2.imwrite(str(out_dir / "puzzle_04_out.jpg"), out4)

# Puzzle 5
img5 = spot_the_differences.load_bgr(str(val_dir / "puzzle_05.jpg"))
img_a, img_b, _, _ = spot_the_differences.auto_slice(img5)
img_b_aligned, valid_y_range, _ = spot_the_differences.align(img_a, img_b, skip_ecc=False)
circles5, _ = spot_the_differences.detect(img_a, img_b_aligned, min_area=50, delta_floor=7.0, valid_y_range=valid_y_range)

out5 = img_a.copy()
for cx, cy, r in circles5:
    cv2.circle(out5, (cx, cy), r, (0, 0, 255), 3)
cv2.imwrite(str(out_dir / "puzzle_05_out.jpg"), out5)

# Puzzle 1
img1 = spot_the_differences.load_bgr(str(val_dir / "puzzle_01.png"))
img_a, img_b, _, _ = spot_the_differences.auto_slice(img1)
img_b_aligned, valid_y_range, _ = spot_the_differences.align(img_a, img_b, skip_ecc=False)
circles1, _ = spot_the_differences.detect(img_a, img_b_aligned, min_area=50, delta_floor=7.0, valid_y_range=valid_y_range)

out1 = img_a.copy()
for cx, cy, r in circles1:
    cv2.circle(out1, (cx, cy), r, (0, 0, 255), 3)
cv2.imwrite(str(out_dir / "puzzle_01_out.png"), out1)

# Puzzle 3
img3 = spot_the_differences.load_bgr(str(val_dir / "puzzle_03.jpg"))
img_a, img_b, _, _ = spot_the_differences.auto_slice(img3)
img_b_aligned, valid_y_range, _ = spot_the_differences.align(img_a, img_b, skip_ecc=False)
circles3, _ = spot_the_differences.detect(img_a, img_b_aligned, min_area=50, delta_floor=7.0, valid_y_range=valid_y_range)

out3 = img_a.copy()
for cx, cy, r in circles3:
    cv2.circle(out3, (cx, cy), r, (0, 0, 255), 3)
cv2.imwrite(str(out_dir / "puzzle_03_out.jpg"), out3)

print("Done debugging puzzles.")
