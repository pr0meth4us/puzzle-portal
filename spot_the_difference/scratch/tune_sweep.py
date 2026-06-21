import cv2
import numpy as np
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

import spot_the_differences

val_dir = SCRIPT_DIR / "validation_dataset"
expected = {
    "puzzle_01.png": 10,
    "puzzle_02.jpg": 10,
    "puzzle_03.jpg": 8,
    "puzzle_04.jpg": 12,
    "puzzle_05.jpg": 10,
    "puzzle_06.jpg": 17
}

# Cache aligned images
cache = {}
print("Pre-aligning images...")
for puzzle_name in expected.keys():
    print(f"Aligning {puzzle_name}...")
    img = spot_the_differences.load_bgr(str(val_dir / puzzle_name))
    img_a, img_b, a_start, b_start = spot_the_differences.auto_slice(img)
    img_b_aligned, valid_y_range, H_align = spot_the_differences.align(img_a, img_b, skip_ecc=False)
    cache[puzzle_name] = {
        "img_a": img_a,
        "img_b_aligned": img_b_aligned,
        "valid_y_range": valid_y_range,
        "H_align": H_align
    }
print("Pre-alignment finished.")

def sweep_colour(puzzle_name, target):
    data = cache[puzzle_name]
    img_a = data["img_a"]
    img_b_aligned = data["img_b_aligned"]
    valid_y_range = data["valid_y_range"]
    
    # Grid search parameters
    delta_floors = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
    min_areas = [10, 20, 30, 40, 50, 75, 100, 150]
    merge_radii = [20, 30, 40, 50, 55, 60, 70, 80]
    
    matching = []
    for df in delta_floors:
        for ma in min_areas:
            for mr in merge_radii:
                # Override parameters
                spot_the_differences.MERGE_RADIUS = mr
                try:
                    circles, count = spot_the_differences.detect(
                        img_a, img_b_aligned, min_area=ma, delta_floor=df, valid_y_range=valid_y_range
                    )
                    if count == target:
                        matching.append((df, ma, mr))
                except Exception:
                    pass
    
    print(f"\n{puzzle_name} (target: {target}): found {len(matching)} configs.")
    if matching:
        print("Sample configurations (delta_floor, min_area, merge_radius):")
        for cfg in matching[:5]:
            print(f"  delta_floor={cfg[0]}, min_area={cfg[1]}, merge_radius={cfg[2]}")
        return matching[0]
    return None

# Sweep for colour puzzles
for puzzle, target in expected.items():
    if puzzle in ["puzzle_01.png", "puzzle_02.jpg", "puzzle_03.jpg", "puzzle_05.jpg"]:
        sweep_colour(puzzle, target)
