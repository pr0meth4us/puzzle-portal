import os
import shutil
from pathlib import Path

BASE_DIR = Path("/Users/nicksng/code/puzzle-portal/spot_the_difference")
SCRATCH_DIR = BASE_DIR / "scratch"

# Files to move to scratch/
files_to_move = [
    "apply_fixes.py",
    "count_red_circles.py",
    "debug_puzzles.py",
    "debug_swan.py",
    "debug_swan_2.py",
    "diff_answers.py",
    "diff_roi.py",
    "extract_truth_puz6.py",
    "find_answer3_dots.py",
    "find_circled_cells.py",
    "find_grid_size.py",
    "find_marker_hues.py",
    "generate_html.py",
    "get_true_counts.py",
    "inspect_all_contours_ans6.py",
    "inspect_answer_color.py",
    "inspect_blue_mask.py",
    "inspect_deltas.py",
    "inspect_green_mask.py",
    "inspect_puzzle6.py",
    "match_answers.py",
    "print_shapes.py",
    "revert_hacks.py",
    "scratch_debug_puz3.py",
    "scratch_ocr.py",
    "scratch_print_old_groups.py",
    "scratch_print_puz5_groups.py",
    "scratch_puz3_candidates.py",
    "scratch_puz5_candidates.py",
    "test_align_puz3.py",
    "test_bbox.py",
    "test_cell_diff.py",
    "test_cell_diff_sliding.py",
    "test_cell_ocr.py",
    "test_kmeans.py",
    "test_puz3_new_mask.py",
    "tune_line.py",
    "tune_sweep.py"
]

# Junk files to delete in BASE_DIR
files_to_delete = [
    "swan_base.jpg",
    "swan_bot_L.jpg",
    "swan_bot_R.jpg",
    "swan_mid_L.jpg",
    "swan_mid_R.jpg",
    "swan_part_0.jpg",
    "swan_part_1.jpg",
    "swan_part_2.jpg",
    "swan_part_3.jpg",
    "swan_part_4.jpg"
]

# Add thumbnails to delete
for i in range(1, 9):
    files_to_delete.append(f"thumb_ans_{i}.jpg")
for i in range(1, 7):
    files_to_delete.append(f"thumb_puz_{i}.jpg")

print("Moving files to scratch...")
for f in files_to_move:
    src = BASE_DIR / f
    dst = SCRATCH_DIR / f
    if src.exists():
        print(f"  Moving {f} -> scratch/")
        shutil.move(src, dst)

print("\nDeleting junk image files...")
for f in files_to_delete:
    p = BASE_DIR / f
    if p.exists():
        print(f"  Deleting {f}")
        p.unlink()

# Delete debug_out directory
debug_out_dir = BASE_DIR / "debug_out"
if debug_out_dir.exists():
    print("\nDeleting debug_out directory...")
    shutil.rmtree(debug_out_dir)

# Clean results directory of untracked intermediate files
results_dir = BASE_DIR / "results"
if results_dir.exists():
    print("\nCleaning results directory...")
    for f in os.listdir(results_dir):
        # Keep only standard validation outputs
        if f not in ["res_puzzle_02.png", "res_puzzle_03.png", "res_puzzle_04.png", 
                     "res_puzzle_05.png", "res_puzzle_06.png", "res_puzzle_07.png", 
                     "res_puzzle_08.png", "res_puzzle_extra_05_vs_puzzle_extra_06.png",
                     ".gitkeep"]:
            p = results_dir / f
            if p.is_file():
                print(f"  Deleting result file: {f}")
                p.unlink()
            elif p.is_dir():
                print(f"  Deleting result dir: {f}")
                shutil.rmtree(p)

print("\nCleanup completed.")
