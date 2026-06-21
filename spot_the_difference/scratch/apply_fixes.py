import sys

with open("spot_the_differences.py", "r") as f:
    code = f.read()

# 1. Update auto_slice
old_auto_slice_end = """    # Compare horizontal vs vertical
    if best_score_h >= best_score_v and best_score_h > 0:
        print(f"[INFO] Auto-sliced (Horizontal) → A{best_a_h.shape[:2]} B{best_b_h.shape[:2]} (SSIM {best_score_h:.2f} >= {best_score_v:.2f})")
        return best_a_h, best_b_h, (0, best_astart_h), (0, best_bstart_h)
    elif best_score_v > best_score_h and best_score_v > 0:
        print(f"[INFO] Auto-sliced (Vertical) → A{best_a_v.shape[:2]} B{best_b_v.shape[:2]} (SSIM {best_score_v:.2f} > {best_score_h:.2f})")
        return best_a_v, best_b_v, (best_astart_v, 0), (best_bstart_v, 0)
    else:
        # Fallback if no peaks found (e.g. solid color)
        half = h // 2
        print("[WARN] No valid separators found! Falling back to horizontal mid-cut.")
        a, b = img[:half], img[half:]
        return a, b, (0, 0), (0, half)"""

if old_auto_slice_end not in code:
    print("Could not find auto_slice end block")
    sys.exit(1)

# 2. Update main()
old_main = """        # Check for Swan Puzzle
        if h == 940 and w == 480:
            print("[INFO] Detected Swan puzzle dimensions. Switching to Swan mode.")
            args.mode = "swan"
            img_a = combined
            img_b = combined
            a_start = b_start = 0
            two_image_mode = False
        else:
            cropped_combined, crop_y_offset = crop_text_by_gap(combined)
            img_a, img_b, a_start, b_start = auto_slice(cropped_combined)
            a_start += crop_y_offset
            b_start += crop_y_offset
            two_image_mode = False
            
    elif len(args.images) == 2:
        print(f"[INFO] Two images: {args.images[0]!r}  vs  {args.images[1]!r}")
        combined       = None
        a_start        = b_start = 0
        img_a          = load_bgr(args.images[0])
        img_b          = load_bgr(args.images[1])
        two_image_mode = True"""

new_main = """        # Check for Swan Puzzle
        if h == 940 and w == 480:
            print("[INFO] Detected Swan puzzle dimensions. Switching to Swan mode.")
            args.mode = "swan"
            img_a = combined
            img_b = combined
            a_loc, b_loc = (0,0), (0,0)
            two_image_mode = False
        else:
            cropped_combined, crop_y_offset = crop_text_by_gap(combined)
            img_a, img_b, a_loc, b_loc = auto_slice(cropped_combined)
            a_loc = (a_loc[0], a_loc[1] + crop_y_offset)
            b_loc = (b_loc[0], b_loc[1] + crop_y_offset)
            two_image_mode = False
            
    elif len(args.images) == 2:
        print(f"[INFO] Two images: {args.images[0]!r}  vs  {args.images[1]!r}")
        combined       = None
        a_loc, b_loc   = (0,0), (0,0)
        img_a          = load_bgr(args.images[0])
        img_b          = load_bgr(args.images[1])
        two_image_mode = True"""
code = code.replace(old_main, new_main)

# 3. Update main() calls to build_stacked_output
old_calls = """        if two_image_mode:
            result = build_sidebyside_output_numgrid(
                img_a, img_b, circles_a, circles_b, color, count)
        else:
            result = build_stacked_output_numgrid(
                combined, img_a, circles_a, circles_b,
                a_start, b_start, color, count)
    else:
        # ── COLOR / LINE DRAWING MODE ─────────────────────────────────────────
        # ...
        if two_image_mode:
            result = build_sidebyside_output(
                img_a, img_b_original, img_b, circles, color, count, H)
        else:
            result = build_stacked_output(
                combined, img_a, img_b, circles,
                a_start, b_start, color, count, H)"""

new_calls = """        if two_image_mode:
            result = build_sidebyside_output_numgrid(
                img_a, img_b, circles_a, circles_b, color, count)
        else:
            result = build_stacked_output_numgrid(
                combined, img_a, circles_a, circles_b,
                a_loc, b_loc, color, count)
    else:
        # ── COLOR / LINE DRAWING MODE ─────────────────────────────────────────
        # ...
        if two_image_mode:
            result = build_sidebyside_output(
                img_a, img_b_original, img_b, circles, color, count, H)
        else:
            result = build_stacked_output(
                combined, img_b_original, circles,
                b_loc, color, count, H)"""

# I'll just use a targeted replace for the exact lines in main().
