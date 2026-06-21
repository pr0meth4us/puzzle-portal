import cv2
import sys
from pathlib import Path

sys.path.append("/Users/nicksng/code/random/spot_the_difference")
import spot_the_differences as std

for puz in ["puzzle_07.jpg", "puzzle_08.jpg"]:
    p_path = Path("/Users/nicksng/code/random/spot_the_difference/puzzles") / puz
    print(f"\n================ {puz} ================")
    combined = std.load_bgr(p_path)
    cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
    img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
    img_b_aligned, valid_y_range, H_align = std.align(img_a, img_b, skip_ecc=False)
    ssim_aligned = std._ssim(img_a, img_b_aligned)
    print(f"Final SSIM after alignment: {ssim_aligned:.4f}")
