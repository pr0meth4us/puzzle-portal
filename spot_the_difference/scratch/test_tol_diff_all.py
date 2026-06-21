import cv2
import numpy as np
import sys
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from spot_the_differences import load_bgr, crop_text_by_gap, auto_slice, align

def _tol_diff(a: np.ndarray, b: np.ndarray, ksize=5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    a_min = cv2.erode(a, kernel)
    a_max = cv2.dilate(a, kernel)
    b_min = cv2.erode(b, kernel)
    b_max = cv2.dilate(b, kernel)
    diff_b_to_a = np.maximum(0, b.astype(float) - a_max.astype(float)) + np.maximum(0, a_min.astype(float) - b.astype(float))
    diff_a_to_b = np.maximum(0, a.astype(float) - b_max.astype(float)) + np.maximum(0, b_min.astype(float) - a.astype(float))
    return np.minimum(diff_b_to_a, diff_a_to_b).astype(np.uint8)

def test_puzzles():
    val_dir = Path("validation_dataset")
    expected = {
        "puzzle_02.jpg": 10,
        "puzzle_03.jpg": 10,
        "puzzle_04.jpg": 12,
        "puzzle_05.jpg": 8,
        "puzzle_06.jpg": 19
    }
    
    # We will run on puzzle_07 to see how many it detects!
    expected["../puzzles/puzzle_07.jpg"] = 12  # we target ~11-13
    expected["../puzzles/puzzle_08.jpg"] = 10
    
    for puzzle, count in expected.items():
        p_path = val_dir / puzzle
        print(f"\n=========================================")
        print(f"Testing {puzzle} (expected: {count})")
        print(f"=========================================")
        combined = load_bgr(str(p_path))
        cropped_combined, crop_y_offset = crop_text_by_gap(combined)
        img_a, img_b, a_start, b_start = auto_slice(cropped_combined)
        
        # Check alignment
        img_b_aligned, valid_y_range, H_align = align(img_a, img_b, skip_ecc=False)
        
        # Convert to gray
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b_aligned, cv2.COLOR_BGR2GRAY)
        
        # Test basic tol_diff stats
        td = _tol_diff(gray_a, gray_b, ksize=5)
        print(f"Tolerance diff mean: {td.mean():.4f}")

if __name__ == "__main__":
    test_puzzles()
