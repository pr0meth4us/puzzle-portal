import cv2
import numpy as np
import sys
sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

puz = cv2.imread("puzzles/puzzle_07.jpg")
cropped_combined, crop_y_offset = std.crop_text_by_gap(puz)
img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
img_b_aligned, _, _ = std.align(img_a, img_b, skip_ecc=False)

# Convert to LAB
la = cv2.cvtColor(img_a, cv2.COLOR_BGR2LAB).astype(np.float32)
lb = cv2.cvtColor(img_b_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)

# Full LAB delta
delta_full = np.sqrt(np.sum((la - lb) ** 2, axis=2))

# Color channels (a, b) delta only
delta_color = np.sqrt(np.sum((la[:, :, 1:] - lb[:, :, 1:]) ** 2, axis=2))

# Convert to HSV
hsva = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV).astype(np.float32)
hsvb = cv2.cvtColor(img_b_aligned, cv2.COLOR_BGR2HSV).astype(np.float32)

# Saturation delta
delta_sat = np.abs(hsva[:, :, 1] - hsvb[:, :, 1])

# Save difference maps normalized to 0-255
def save_norm(name, img):
    mx = img.max()
    norm = (img / mx * 255).astype(np.uint8) if mx > 0 else img.astype(np.uint8)
    cv2.imwrite(f"/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/{name}.png", norm)

save_norm("diff_full", delta_full)
save_norm("diff_color", delta_color)
save_norm("diff_sat", delta_sat)

print("Saved difference maps to artifacts directory.")
