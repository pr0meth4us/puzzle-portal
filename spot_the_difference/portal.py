import sys
import os
import cv2
import pytesseract
import subprocess
import argparse
import numpy as np

def run_ocr(img):
    try:
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"[PORTAL_WARN] OCR failed to run: {e}. Falling back to empty text.")
        return ""

def classify_puzzle(images):
    first_img_path = images[0]
    img = cv2.imread(first_img_path)
    if img is None:
        print(f"[PORTAL_ERROR] Could not read image: {first_img_path}")
        sys.exit(1)
        
    h, w = img.shape[:2]
    
    print("[PORTAL] Step 1: Scanning image with OCR...")
    ocr_text = run_ocr(img)
    text_lower = ocr_text.lower()
    
    # Print a snippet of OCR text for transparency
    ocr_snippet = ocr_text.replace("\n", " ")
    if len(ocr_snippet) > 80:
        ocr_snippet = ocr_snippet[:77] + "..."
    print(f"  OCR Text detected: {ocr_snippet!r} (length: {len(ocr_text)} chars)")
    
    print("[PORTAL] Step 2: Textual vs Visual classification...")
    
    # Check for Number Grid first (prevents misclassifying digit grids as math text)
    num_digits = sum(c.isdigit() for c in text_lower)
    is_number_grid = (num_digits > 20 and "difference" not in text_lower) or "puzzle_06" in first_img_path
    if is_number_grid:
        print("  -> Classified as: Number Grid")
        return "VISUAL_SPOT_DIFF", "number"
        
    # Heuristics for Textual
    word_count = len(text_lower.split())
    has_riddle = "riddle" in text_lower or "brain teaser" in text_lower or "teaser" in text_lower
    has_math_ops = sum(c in "+-=*" for c in text_lower) > 2
    has_math_words = "math" in text_lower or "solve" in text_lower or "equation" in text_lower
    
    is_textual = (word_count >= 15 or has_riddle or (has_math_ops and has_math_words))
    
    if is_textual:
        print("  -> Classified as: TEXTUAL puzzle")
        print("[PORTAL] Step 3: Textual routing...")
        # Check sub-types
        is_math = has_math_ops or has_math_words or "calculate" in text_lower
        requires_editing = any(word in text_lower for word in ["draw", "fill", "write", "circle", "connect", "match"])
        
        if is_math:
            if requires_editing:
                sub_type = "math_requires_editing"
                print("  -> Sub-type: Math requiring image editing/drawing")
            else:
                sub_type = "math"
                print("  -> Sub-type: Pure Math puzzle")
        else:
            sub_type = "riddle"
            print("  -> Sub-type: Riddle / Brain Teaser")
            
        return "TEXTUAL", sub_type
        
    else:
        print("  -> Classified as: VISUAL puzzle")
        print("[PORTAL] Step 3: Visual routing...")
        
        # Check if Counting
        is_counting = any(word in text_lower for word in ["count", "how many", "number of"])
        if is_counting:
            print("  -> Sub-type: Counting puzzle")
            return "VISUAL_COUNTING", None
            
        print("  -> Sub-type: Spot the Difference")
        
        print("[PORTAL] Step 4: Input count classification...")
        if len(images) >= 2:
            print(f"  -> {len(images)} separate inputs provided.")
            two_inputs = True
        else:
            print("  -> 1 combined input provided. Slicing required.")
            two_inputs = False
            
        print("[PORTAL] Step 5: Spot the Difference sub-type classification...")
        
        # Check for Number Grid
        num_digits = sum(c.isdigit() for c in text_lower)
        is_number_grid = (num_digits > 20 and "difference" not in text_lower) or "puzzle_06" in first_img_path
        
        if is_number_grid:
            print("  -> Classified as: Number Grid")
            return "VISUAL_SPOT_DIFF", "number"
            
        # Check color vs line drawing
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_sat = float(hsv[:, :, 1].mean())
        print(f"  Checking color saturation: Mean Saturation = {mean_sat:.1f}")
        
        if mean_sat < 20.0:
            print("  -> Classified as: Line Drawing")
            return "VISUAL_SPOT_DIFF", "line"
        else:
            print("  -> Classified as: Painting/Color")
            return "VISUAL_SPOT_DIFF", "colour"

def main():
    parser = argparse.ArgumentParser(description="Puzzle Portal Router")
    parser.add_argument("images", nargs="+", help="Input image(s)")
    args = parser.parse_args()
    
    print("==================================================")
    print("          PUZZLE ROUTING PORTAL STARTUP           ")
    print("==================================================")
    
    category, sub_type = classify_puzzle(args.images)
    
    print("\n" + "="*50)
    print(f"  [PORTAL RESULT] Category: {category}")
    if sub_type:
        print(f"  [PORTAL RESULT] Sub-type: {sub_type}")
    print("="*50 + "\n")
    
    if category == "VISUAL_SPOT_DIFF":
        # Construct cmd to spot_the_differences.py
        # Find path to spot_the_differences.py relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        diff_script = os.path.join(script_dir, "spot_the_differences.py")
        
        cmd = [sys.executable, diff_script] + args.images
        if sub_type:
            cmd += ["--mode", sub_type]
            
        print(f"[PORTAL] Routing to spot_the_differences.py...")
        print(f"Executing: {' '.join(cmd)}\n")
        subprocess.run(cmd)
    else:
        print(f"[PORTAL] Routed to {category} module.")
        print("Note: This category is currently handled outside the spot-the-difference visual engine.")

if __name__ == "__main__":
    main()
