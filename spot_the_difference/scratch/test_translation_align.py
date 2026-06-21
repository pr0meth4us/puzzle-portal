import cv2
import numpy as np
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SCRIPT_DIR))
import spot_the_differences as std

def run_translation_test(puz_name):
    print(f"\n=========================================")
    print(f"Testing translation-only alignment on {puz_name}")
    print(f"=========================================")
    p_path = SCRIPT_DIR / "puzzles" / puz_name
    combined = cv2.imread(str(p_path))
    
    cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
    img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
    
    # Run SIFT matching
    g_ref = std._gray(img_a)
    g_tgt = std._gray(img_b)
    kp1, kp2, good = std._match(g_ref, g_tgt, "SIFT")
    
    src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    # Calculate median translation
    dx = float(np.median(dst[:, 0, 0] - src[:, 0, 0]))
    dy = float(np.median(dst[:, 0, 1] - src[:, 0, 1]))
    print(f"SIFT matches: {len(good)} | Median translation: dx={dx:.3f}, dy={dy:.3f}")
    
    # Build translation-only Homography matrix
    H_trans = np.array([[1.0, 0.0, dx],
                        [0.0, 1.0, dy],
                        [0.0, 0.0, 1.0]], dtype=np.float32)
    
    h, w = img_a.shape[:2]
    img_b_aligned = std._warp_h(img_b, H_trans, (w, h))
    
    s_before = std._ssim(img_a, img_b)
    s_after = std._ssim(img_a, img_b_aligned)
    print(f"SSIM before: {s_before:.4f} | SSIM after translation align: {s_after:.4f}")
    
    # Run detect
    is_vertical_split = (img_a.shape[0] == cropped_combined.shape[0])
    split_dir = "vertical" if is_vertical_split else "horizontal"
    
    circles, count = std.detect(img_a, img_b_aligned, min_area=30, split_dir=split_dir, H=H_trans)
    print(f"Differences found: {count}")
    for i, c in enumerate(circles):
        print(f"  Circle {i+1}: {c}")

run_translation_test("puzzle_07.jpg")
run_translation_test("puzzle_08.jpg")
