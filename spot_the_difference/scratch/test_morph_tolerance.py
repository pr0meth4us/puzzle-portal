import cv2
import numpy as np
import sys
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from spot_the_differences import load_bgr, crop_text_by_gap, auto_slice, align

def main():
    combined = load_bgr("puzzles/puzzle_07.jpg")
    cropped_combined, crop_y_offset = crop_text_by_gap(combined)
    img_a, img_b, a_start, b_start = auto_slice(cropped_combined)
    
    img_b_aligned, valid_y_range, H_align = align(img_a, img_b, skip_ecc=False)
    
    h, w = img_a.shape[:2]
    
    # Let's convert to grayscale
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b_aligned, cv2.COLOR_BGR2GRAY)
    
    # 1. Standard absolute difference
    abs_diff = cv2.absdiff(gray_a, gray_b)
    
    # 2. Morphological tolerance difference
    # Tolerance of 2 pixels -> 5x5 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # Erode and dilate image A
    a_min = cv2.erode(gray_a, kernel)
    a_max = cv2.dilate(gray_a, kernel)
    
    # Erode and dilate image B
    b_min = cv2.erode(gray_b, kernel)
    b_max = cv2.dilate(gray_b, kernel)
    
    # Tolerance difference
    diff_b_to_a = np.maximum(0, gray_b.astype(float) - a_max.astype(float)) + np.maximum(0, a_min.astype(float) - gray_b.astype(float))
    diff_a_to_b = np.maximum(0, gray_a.astype(float) - b_max.astype(float)) + np.maximum(0, b_min.astype(float) - gray_a.astype(float))
    
    # Symmetric tolerance difference
    tol_diff = np.minimum(diff_b_to_a, diff_a_to_b).astype(np.uint8)
    
    # Print statistics
    print(f"Standard abs_diff mean: {abs_diff.mean():.4f}")
    print(f"Tolerance diff mean: {tol_diff.mean():.4f}")
    
    # Save the difference images to inspect
    cv2.imwrite("results/debug_abs_diff.png", abs_diff)
    cv2.imwrite("results/debug_tol_diff.png", tol_diff)
    print("Saved debug_abs_diff.png and debug_tol_diff.png")

if __name__ == "__main__":
    main()
