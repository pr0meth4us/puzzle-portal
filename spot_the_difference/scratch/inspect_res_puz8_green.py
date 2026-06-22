import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference")
sys.path.append(str(SCRIPT_DIR.parent))
import spot_the_difference.spot_the_differences as std

img_path = SCRIPT_DIR / "puzzles" / "puzzle_08.jpg"
combined = std.load_bgr(str(img_path))
cropped, crop_y_offset = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped)

print("Aligning...")
img_b_aligned, valid_y_range, H_align, valid_mask = std.align(img_a, img_b, skip_ecc=False)

print("Detecting...")
circles, count = std.detect(img_a, img_b_aligned,
                            min_area=50,
                            delta_floor=12.0,
                            valid_y_range=valid_y_range,
                            split_dir="horizontal",
                            H=H_align,
                            edge_mask_ksize=5,
                            merge_radius_override=40,
                            valid_mask=valid_mask)

# Force color to Green (0, 255, 0)
color = (0, 255, 0)

# Build output
H_inv = np.linalg.inv(H_align) if H_align is not None else None
ph = img_a.shape[0]
b_slice = combined[b_start:b_start + ph, :]
pil_b = Image.fromarray(cv2.cvtColor(b_slice, cv2.COLOR_BGR2RGB))

# Call drawing function
pil_b = std.draw_contours_and_numbers_on_panel(pil_b, circles, img_a, img_b_aligned, H_inv, color, valid_mask)

# Save test output
output_path = SCRIPT_DIR / "scratch" / "test_puz8_green.png"
pil_b.save(output_path)

# Let's inspect the shapes drawn in pure green in the saved image
img_test = cv2.imread(str(output_path))
hsv = cv2.cvtColor(img_test, cv2.COLOR_BGR2HSV)
# Pure green in HSV has H around 60 (120 degrees), S=255, V=255
mask = cv2.inRange(hsv, (55, 200, 200), (65, 255, 255))
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"\nAnalyzing shapes drawn in pure green:")
circle_count = 0
contour_count = 0
for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    if area < 10:
        continue
    peri = cv2.arcLength(c, True)
    circ = 4 * np.pi * area / (peri ** 2) if peri > 0 else 0
    (cx, cy), r = cv2.minEnclosingCircle(c)
    
    # Check if circularity is high (perfect circle has circularity ~ 1.0)
    if circ > 0.85:
        shape_type = "CIRCLE"
        circle_count += 1
    else:
        shape_type = "CONTOUR"
        contour_count += 1
    print(f"  Shape {i+1}: Center=({cx:.1f}, {cy:.1f}), Radius={r:.1f}, Area={area:.1f}, Circularity={circ:.3f} -> {shape_type}")

print(f"\nSummary: {circle_count} CIRCLES, {contour_count} CONTOURS")
