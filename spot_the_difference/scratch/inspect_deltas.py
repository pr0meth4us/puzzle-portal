import cv2
import numpy as np
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

import spot_the_differences

val_dir = SCRIPT_DIR / "validation_dataset"
puzzles = [
    "puzzle_01.png",
    "puzzle_02.jpg",
    "puzzle_03.jpg",
    "puzzle_04.jpg",
    "puzzle_05.jpg",
    "puzzle_06.jpg"
]

for puzzle in puzzles:
    print(f"\n=========================================")
    print(f"Inspecting {puzzle}")
    print(f"=========================================")
    
    img = spot_the_differences.load_bgr(str(val_dir / puzzle))
    targets = {
        "puzzle_01.png": 10,
        "puzzle_02.jpg": 10,
        "puzzle_03.jpg": 10,
        "puzzle_04.jpg": 11,
        "puzzle_05.jpg": 7,
        "puzzle_06.jpg": 30
    }
    
    target = targets[puzzle]
    print(f"Target count: {target}")

    # Fix auto-slice for Puzzle 3 (Elephant) which fails SSIM due to white margins
    if puzzle == "puzzle_03.jpg":
        sep = img.shape[0] // 2
        img_a, img_b = img[:sep, :], img[sep:, :]
        print("[INFO] Auto-sliced manually (Horizontal) for Puzzle 3")
    else:
        img_a, img_b, a_start, b_start = spot_the_differences.auto_slice(img)
        
    img_b_aligned, valid_y_range, H_align = spot_the_differences.align(img_a, img_b, skip_ecc=False)
    
    mean_sat = float(cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
    print(f"Mean saturation: {mean_sat:.1f}")
    
    # Run detection
    if puzzle == "puzzle_04.jpg":
        # Line mode
        circles, count = spot_the_differences.detect_line(img_a, img_b_aligned)
        print(f"Line mode - Found {count} circles:")
        for i, c in enumerate(circles):
            print(f"  Circle {i+1}: cx={c[0]}, cy={c[1]}, r={c[2]}")
        if count == target:
            print(f"  --> MATCH TARGET!")
    elif puzzle == "puzzle_06.jpg":
        try:
            circles_a, circles_b, count, _ca, _cb = spot_the_differences.detect_number_grid(img_a, img_b)
            print(f"OCR mode - Found {count} differences")
            if count == target:
                print(f"  --> MATCH TARGET!")
        except Exception as e:
            print("OCR mode error:", e)
    else:
        # Colour mode
        print("Running color detection with delta_floor=7.0:")
        
        # Apply specific crops to ignore text
        if puzzle == "puzzle_01.png":
            # Mask out the text "FIND THE 5 DIFFERENCES" at the bottom
            valid_y_range = (0, int(img_a.shape[0] * 0.88))
        
        circles, count = spot_the_differences.detect(img_a, img_b_aligned, min_area=50, delta_floor=7.0, valid_y_range=valid_y_range)
        print(f"Default run found: {count} differences")
        
        if count == target:
            print(f"  --> MATCH TARGET WITH delta_floor=7.0!")
        else:
            # Let's run with different floors to see where count hits target
            for df in [5.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]:
                circles, count = spot_the_differences.detect(img_a, img_b_aligned, min_area=50, delta_floor=df, valid_y_range=valid_y_range)
                if count == target:
                    print(f"  --> MATCH TARGET WITH delta_floor={df}!")
                    break
