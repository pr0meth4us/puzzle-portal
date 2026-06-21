import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from spot_the_differences import load_bgr, align, detect, _gray

def main():
    img_path = Path("puzzles/puzzle_07.jpg")
    combined = load_bgr(str(img_path))
    h, w = combined.shape[:2]
    
    # Slicing
    from spot_the_differences import crop_text_by_gap, auto_slice
    cropped_combined, crop_y_offset = crop_text_by_gap(combined)
    img_a, img_b, a_start, b_start = auto_slice(cropped_combined)
    is_vertical_split = (img_a.shape[0] == cropped_combined.shape[0])
    
    img_b_aligned, valid_y_range, H_align = align(img_a, img_b, skip_ecc=False)
    
    split_dir = "vertical" if is_vertical_split else "horizontal"
    
    # Let's run detect
    circles, count = detect(img_a, img_b_aligned,
                            min_area=30,
                            delta_floor=7.0,
                            valid_y_range=valid_y_range,
                            split_dir=split_dir,
                            H=H_align)
                            
    print(f"Detected {count} differences:")
    for idx, (cx, cy, r) in enumerate(circles):
        print(f"Circle {idx}: center=({cx}, {cy}), r={r}")
        
    # Draw labeled circles on img_a
    img_debug = img_a.copy()
    for idx, (cx, cy, r) in enumerate(circles):
        cv2.circle(img_debug, (cx, cy), r, (0, 0, 255), 2)
        cv2.putText(img_debug, str(idx), (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    cv2.imwrite("results/debug_puz07_labeled.png", img_debug)
    print("Saved results/debug_puz07_labeled.png")

if __name__ == "__main__":
    main()
