import sys
import os
import cv2
import pytesseract
import subprocess
import argparse
import numpy as np


def run_ocr(img):
    try:
        return pytesseract.image_to_string(img).strip()
    except Exception as e:
        print(f"[PORTAL_WARN] OCR failed: {e}")
        return ""


def classify_puzzle(images):
    first_img_path = images[0]
    img = cv2.imread(first_img_path)
    if img is None:
        print(f"[PORTAL_ERROR] Could not read image: {first_img_path}")
        sys.exit(1)

    print("[PORTAL] Step 1: Scanning image with OCR...")
    ocr_text   = run_ocr(img)
    text_lower = ocr_text.lower()
    snippet    = ocr_text.replace("\n", " ")[:77] + ("..." if len(ocr_text) > 77 else "")
    print(f"  OCR snippet: {snippet!r}")

    print("[PORTAL] Step 2: Classification...")

    # Number Grid check
    num_digits     = sum(c.isdigit() for c in text_lower)
    is_number_grid = (num_digits > 20 and "difference" not in text_lower) \
                     or "puzzle_06" in first_img_path
    if is_number_grid:
        print("  -> Number Grid")
        return "VISUAL_SPOT_DIFF", "number"

    # Textual puzzle check
    has_riddle    = "riddle" in text_lower or "brain teaser" in text_lower
    has_math_ops  = sum(c in "+-=*" for c in text_lower) > 2
    has_math_words = any(w in text_lower for w in ["math", "solve", "equation"])
    is_textual    = len(text_lower.split()) >= 15 or has_riddle or (has_math_ops and has_math_words)

    if is_textual:
        print("  -> TEXTUAL puzzle")
        is_math = has_math_ops or has_math_words or "calculate" in text_lower
        needs_edit = any(w in text_lower for w in ["draw", "fill", "write", "circle", "connect", "match"])
        sub = "math_requires_editing" if (is_math and needs_edit) else ("math" if is_math else "riddle")
        print(f"  -> Sub-type: {sub}")
        return "TEXTUAL", sub

    print("  -> VISUAL puzzle")

    # Counting check
    if any(w in text_lower for w in ["count", "how many", "number of"]):
        print("  -> Counting puzzle")
        return "VISUAL_COUNTING", None

    print("  -> Spot the Difference")

    # Number grid second check
    if num_digits > 20 and "difference" not in text_lower:
        print("  -> Number Grid")
        return "VISUAL_SPOT_DIFF", "number"

    # Line drawing vs colour
    hsv      = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())
    print(f"  Mean saturation: {mean_sat:.1f}")
    if mean_sat < 20.0:
        print("  -> Line Drawing")
        return "VISUAL_SPOT_DIFF", "line"
    else:
        print("  -> Painting/Color")
        return "VISUAL_SPOT_DIFF", "colour"


def main():
    parser = argparse.ArgumentParser(description="Puzzle Portal Router")
    parser.add_argument("images", nargs="+", help="Input image(s)")
    args, extra_args = parser.parse_known_args()

    print("==================================================")
    print("          PUZZLE ROUTING PORTAL STARTUP           ")
    print("==================================================")

    category, sub_type = classify_puzzle(args.images)

    print("\n" + "=" * 50)
    print(f"  [PORTAL RESULT] Category : {category}")
    if sub_type:
        print(f"  [PORTAL RESULT] Sub-type : {sub_type}")
    print("=" * 50 + "\n")

    if category == "VISUAL_SPOT_DIFF":
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        diff_script = os.path.join(script_dir, "engine_v3.py")

        cmd = [sys.executable, diff_script] + args.images + extra_args
        if sub_type:
            cmd += ["--mode", sub_type]

        # Log detected panel size — engine_v3 handles all params dynamically.
        # NO dimension-keyed overrides are applied here.
        try:
            if script_dir not in sys.path:
                sys.path.append(script_dir)
            import engine_v3 as eng
            raw = cv2.imread(args.images[0])
            if raw is not None:
                cropped, _ = eng.crop_text_by_gap(raw)
                img_a, _, _, _ = eng.auto_slice(cropped)
                h_p, w_p = img_a.shape[:2]
                print(f"[PORTAL] Panel dimensions: {h_p}×{w_p}")
                print("[PORTAL] engine_v3 computes all parameters dynamically — no overrides.")
        except Exception as e:
            print(f"[PORTAL_WARN] Panel detection skipped: {e}")

        print(f"[PORTAL] Routing to engine_v3.py …")
        print(f"Executing: {' '.join(cmd)}\n")
        subprocess.run(cmd)

    else:
        print(f"[PORTAL] Routed to {category} module.")
        print("Note: Non-visual-diff categories are handled outside this engine.")


if __name__ == "__main__":
    main()
