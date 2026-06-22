import sys
from pathlib import Path
import cv2
import numpy as np

SCRIPT_DIR = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference")
sys.path.append(str(SCRIPT_DIR.parent))
import spot_the_difference.spot_the_differences as std

truth_actual = [
    (456.9, 201.1),   # 1. dam top-left rock
    (602.3, 512.4),   # 2. dam middle
    (216.4, 572.8),   # 3. water left
    (404.4, 809.4),   # 4. dam lower
    (948.1, 814.5),   # 5. dam right
    (391.3, 819.0),   # 6. dam lower-left
    (636.2, 864.9),   # 7. dam lower-middle
    (270.9, 923.6),   # 8. dam rock
    (827.2, 907.0),   # 9. dam lower-right
    (72.5, 442.0),    # 10. river patch left
    (71.8, 534.6)     # 11. rock left
]

img_path = SCRIPT_DIR / "puzzles" / "puzzle_07.jpg"
combined = std.load_bgr(str(img_path))
cropped, crop_y_offset = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped)

print("Aligning...")
img_b_aligned, valid_y_range, H_align, valid_mask = std.align(img_a, img_b, skip_ecc=False)

# Let's run detect with refined parameters:
# We will simulate the removal of the w_panel - 180 bypass in OCR text masking by manually defining it
# or running standard detect but with additional mask_rois for the right edge text if we want,
# but let's actually just modify spot_the_differences.py to remove the bypass.
# In this test script, we will run the detection and manually mask the right edge text (x > 980)
# to see if it works, and we will also apply top/bottom/left/right margins.
print("Detecting with refined parameters...")
circles, count = std.detect(img_a, img_b_aligned,
                            min_area=30,
                            delta_floor=12.0, # Let's try delta_floor = 12.0
                            valid_y_range=valid_y_range,
                            split_dir="horizontal",
                            H=H_align,
                            edge_mask_ksize=5,
                            custom_left_margin=48,
                            custom_right_margin=35,
                            custom_top_margin=35,
                            custom_bottom_margin=35,
                            mask_rois=[(810,630,860,680), (880,860,930,910), (980,0,1152,1006)],
                            merge_radius_override=55,
                            valid_mask=valid_mask)

print(f"\nFound {len(circles)} circles:")
matched_gt = set()
fps = 0
for idx, (cx, cy, r) in enumerate(circles):
    closest_dist = float('inf')
    closest_gt = -1
    for i, (gtx, gty) in enumerate(truth_actual):
        dist = np.sqrt((cx - gtx)**2 + (cy - gty)**2)
        if dist < closest_dist:
            closest_dist = dist
            closest_gt = i
            
    status = "FALSE POSITIVE"
    if closest_dist < 60:
        status = f"MATCHED GT {closest_gt+1}"
        matched_gt.add(closest_gt)
    else:
        fps += 1
    print(f"  Circle {idx+1}: Center=({cx}, {cy}), r={r} -> {status} (dist={closest_dist:.1f})")

print(f"\nSummary: Matched GTs: {len(matched_gt)} / {len(truth_actual)}. False Positives: {fps}")
