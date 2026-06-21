import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SCRIPT_DIR))
import spot_the_differences as std

def run_test(puz_name):
    print(f"\n=========================================")
    print(f"Testing margins and OCR on {puz_name}")
    print(f"=========================================")
    p_path = SCRIPT_DIR / "puzzles" / puz_name
    combined = cv2.imread(str(p_path))
    
    cropped_combined, crop_y_offset = std.crop_text_by_gap(combined)
    img_a, img_b, a_start, b_start = std.auto_slice(cropped_combined)
    is_vertical_split = (img_a.shape[0] == cropped_combined.shape[0])
    split_dir = "vertical" if is_vertical_split else "horizontal"
    
    img_b_aligned, valid_y_range, H_align = std.align(img_a, img_b, skip_ecc=False)
    
    # 1. Standard bmask setup
    h, w = img_a.shape[:2]
    border_val = max(16, int(min(h, w) * 0.02))
    bmask = np.zeros_like(_gray(img_a))
    
    if split_dir == "vertical":
        by_top, by_bot = 12, 20
        bmask[by_top:h - by_bot, 16:w - 10] = 255
    elif split_dir == "horizontal":
        by_top, by_bot = 16, 8
        bmask[by_top:h - by_bot, 8:w - 8] = 255
    else:
        by_top, by_bot = border_val, border_val
        bmask[by_top:h - by_bot, border_val:w - border_val] = 255
        
    if H_align is not None:
        mask_b = np.ones(img_b.shape[:2], dtype=np.uint8) * 255
        warped_mask_b = cv2.warpPerspective(mask_b, H_align, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        warped_mask_b = cv2.erode(warped_mask_b, kernel)
        bmask = cv2.bitwise_and(bmask, warped_mask_b)
        
    if valid_y_range:
        vy0, vy1 = valid_y_range
        if vy0 > by_top: bmask[:vy0, :] = 0
        if vy1 < h - by_bot: bmask[vy1:, :] = 0
        
    # 2. Margin masking
    std._mask_margins(img_a, bmask)
    
    # 3. Standard OCR masking
    std._mask_ocr_text(img_a, img_b_aligned, bmask)
    
    # 4. Custom Rotated Color Mask OCR for Vertical Labels on the right side
    try:
        import pytesseract
        crop_w = 150
        for panel, name in [(img_a, "img_a"), (img_b_aligned, "img_b")]:
            crop = panel[:, w - crop_w:]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # Yellow and Red masks
            mask_y = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
            mask_r1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
            mask_r2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
            mask_r = mask_r1 | mask_r2
            
            for mask_type, mask in [("yellow", mask_y), ("red", mask_r)]:
                # Rotate CCW
                rot_mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cfg = "-l eng+khm --psm 11"
                data = pytesseract.image_to_data(Image.fromarray(rot_mask), config=cfg, output_type=pytesseract.Output.DICT)
                n_boxes = len(data.get('level', []))
                for i in range(n_boxes):
                    text = data['text'][i].strip()
                    if text:
                        rx = data['left'][i]
                        ry = data['top'][i]
                        rw = data['width'][i]
                        rh = data['height'][i]
                        
                        # Map back to original panel coordinates
                        # CCW Rotation maps (rx, ry) in (crop_w, h_panel) rotated to:
                        # y_orig = rx
                        # x_orig = crop_w - ry - rh
                        x_orig = crop_w - ry - rh
                        y_orig = rx
                        
                        # Full panel coordinate
                        x_full = w - crop_w + x_orig
                        y_full = y_orig
                        w_full = rh
                        h_full = rw
                        
                        pad = 12 # slightly larger pad for vertical text to cover drop shadow
                        x1 = max(0, x_full - pad)
                        y1 = max(0, y_full - pad)
                        x2 = min(w, x_full + w_full + pad)
                        y2 = min(h, y_full + h_full + pad)
                        
                        bmask[y1:y2, x1:x2] = 0
                        print(f"[TEST_INFO] Masked vertical label: '{text}' at ({x1},{y1})->({x2},{y2})")
    except Exception as e:
        print(f"Custom vertical OCR failed: {e}")
        
    # Run detect
    circles, count = std.detect(img_a, img_b_aligned, min_area=30, split_dir=split_dir, H=H_align)
    # Filter circles that are completely inside masked areas of bmask
    valid_circles = []
    for cx, cy, r in circles:
        # Check if the circle is masked out (value 0 in bmask)
        mask_val = bmask[cy, cx]
        if mask_val > 0:
            valid_circles.append((cx, cy, r))
            
    print(f"Found {len(valid_circles)} differences (filtered from {count}).")
    for i, c in enumerate(valid_circles):
        print(f"  Circle {i+1}: {c}")

def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

run_test("puzzle_07.jpg")
