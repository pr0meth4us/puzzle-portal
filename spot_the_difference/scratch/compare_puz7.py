import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from spot_the_differences import load_bgr, crop_text_by_gap, auto_slice, align, detect

def get_truth_circles():
    img = cv2.imread("correct_answers/answer_07.jpg")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 150, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 150, 50])
    upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for c in contours:
        if cv2.contourArea(c) > 10:
            (cx, cy), r = cv2.minEnclosingCircle(c)
            circles.append((cx, cy, r))
    return circles

def main():
    combined = load_bgr("puzzles/puzzle_07.jpg")
    h_orig, w_orig = combined.shape[:2]
    
    # Scale factors from answer_07 (853x480) to puzzle_07
    ans_img = cv2.imread("correct_answers/answer_07.jpg")
    h_ans, w_ans = ans_img.shape[:2]
    
    scale_x = w_orig / w_ans
    scale_y = h_orig / h_ans
    
    print(f"Original image size: {w_orig}x{h_orig}")
    print(f"Answer image size: {w_ans}x{h_ans}")
    print(f"Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
    
    # Slicing/cropping
    cropped_combined, crop_y_offset = crop_text_by_gap(combined)
    img_a, img_b, a_start, b_start = auto_slice(cropped_combined)
    is_vertical_split = (img_a.shape[0] == cropped_combined.shape[0])
    
    img_b_aligned, valid_y_range, H_align = align(img_a, img_b, skip_ecc=False)
    split_dir = "vertical" if is_vertical_split else "horizontal"
    
    # Run detect
    detected_circles, count = detect(img_a, img_b_aligned,
                                     min_area=30,
                                     delta_floor=7.0,
                                     valid_y_range=valid_y_range,
                                     split_dir=split_dir,
                                     H=H_align)
                                     
    # Translate truth circles to img_a coordinates
    truth_raw = get_truth_circles()
    truth_in_a = []
    truth_in_b = []
    
    for cx, cy, r in truth_raw:
        # Scale to original combined coordinates
        orig_x = cx * scale_x
        orig_y = cy * scale_y
        
        # Now check if it's in panel A or panel B
        # Recall that panel A is the top panel, panel B is the bottom panel
        # crop_y_offset is subtracted if it's horizontal split
        if not is_vertical_split:
            y_in_cropped = orig_y - crop_y_offset
            # img_a is from y=0 to img_a.shape[0] in cropped_combined
            # img_b is from b_start to b_start + img_b.shape[0]
            # Since img_a and img_b are pre-aligned to img_a,
            # we want to find its coordinate relative to the game panel.
            if y_in_cropped < img_a.shape[0]:
                truth_in_a.append((orig_x, y_in_cropped, r * scale_x))
            else:
                y_in_b = y_in_cropped - b_start
                truth_in_b.append((orig_x, y_in_b, r * scale_x))
        else:
            # Vertical split
            # img_a is left, img_b is right
            if orig_x < img_a.shape[1]:
                truth_in_a.append((orig_x, orig_y, r * scale_x))
            else:
                x_in_b = orig_x - b_start
                truth_in_b.append((x_in_b, orig_y, r * scale_x))
                
    print("\n--- GROUND TRUTH CIRCLES IN PANEL A ---")
    for idx, (tx, ty, tr) in enumerate(truth_in_a):
        print(f"Truth A {idx}: ({tx:.1f}, {ty:.1f}), r={tr:.1f}")
        
    print("\n--- GROUND TRUTH CIRCLES IN PANEL B ---")
    for idx, (tx, ty, tr) in enumerate(truth_in_b):
        print(f"Truth B {idx}: ({tx:.1f}, {ty:.1f}), r={tr:.1f}")
        
    print("\n--- DETECTED CIRCLES ---")
    for idx, (dx, dy, dr) in enumerate(detected_circles):
        # Find closest truth circle in A
        closest_dist = float('inf')
        closest_idx = -1
        for tidx, (tx, ty, tr) in enumerate(truth_in_a):
            dist = np.sqrt((dx - tx)**2 + (dy - ty)**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = tidx
        
        # Let's also check if it matches panel B (some puzzles might only have answers on one panel,
        # but here we detect differences between A and B, so a difference in A is also a difference in B)
        closest_dist_b = float('inf')
        closest_idx_b = -1
        for tidx, (tx, ty, tr) in enumerate(truth_in_b):
            dist = np.sqrt((dx - tx)**2 + (dy - ty)**2)
            if dist < closest_dist_b:
                closest_dist_b = dist
                closest_idx_b = tidx
                
        print(f"Detected {idx}: ({dx}, {dy}), r={dr} | closest truth A: {closest_idx} (dist={closest_dist:.1f}), B: {closest_idx_b} (dist={closest_dist_b:.1f})")

if __name__ == "__main__":
    main()
