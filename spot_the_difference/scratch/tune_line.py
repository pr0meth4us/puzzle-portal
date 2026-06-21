import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
val_dir = SCRIPT_DIR / "validation_dataset"

img = cv2.imread(str(val_dir / "puzzle_04.jpg"))
h, w = img.shape[:2]
sep = w // 2 # Auto-slice vertical
img_a, img_b = img[:, :sep], img[:, sep:]

# Let's align
import spot_the_differences
img_b_aligned, valid_y_range, H_align = spot_the_differences.align(img_a, img_b, skip_ecc=False)

def test_detect_line(min_area, delta_floor, max_diff_min, nms_radius):
    h, w  = img_a.shape[:2]
    img_b_resized = cv2.resize(img_b_aligned, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    # SSIM difference
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b_resized, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(gray_a, gray_b, full=True)
    inv = cv2.bitwise_not((diff * 255).clip(0, 255).astype(np.uint8))
    
    _, thresh = cv2.threshold(inv, 30, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k)
    
    border = max(8, int(min(h, w) * 0.02))
    bmask  = np.zeros_like(thresh)
    bmask[border:h - border, border:w - border] = 255
    thresh = cv2.bitwise_and(thresh, bmask)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cdiff = np.mean(cv2.absdiff(img_a, img_b_resized).astype(np.float32), axis=2)
    
    candidates = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
        delta = cv2.mean(cdiff, mask=mask)[0]
        if delta < delta_floor:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        cx, cy, r   = int(cx), int(cy), int(r)
        
        x1 = max(0, cx-r-5)
        y1 = max(0, cy-r-5)
        x2 = min(w, cx+r+5)
        y2 = min(h, cy+r+5)
        
        # Calculate peak pixel difference in local neighborhood
        peak = float(np.abs(gray_a[y1:y2, x1:x2].astype(np.float64) - gray_b[y1:y2, x1:x2].astype(np.float64)).max())
        if peak < max_diff_min:
            continue
        
        max_r = int(min(h, w) * 0.15)
        candidates.append((cx, cy, min(max(r + 15, 20), max_r), delta))
        
    candidates.sort(key=lambda x: -x[3])
    kept = []
    for cx, cy, r, d in candidates:
        if not any(((cx-kx)**2+(cy-ky)**2)**0.5 < nms_radius
                   for kx, ky, _, _ in kept):
            kept.append((cx, cy, r, d))
            
    return len(kept)

# Run sweep
matching_configs = []
for ma in [10, 20, 30, 40, 50]:
    for df in [5.0, 7.0, 10.0, 15.0, 20.0]:
        for mdm in [80, 100, 120, 140, 160]:
            for nr in [25, 30, 35, 40, 45, 50]:
                count = test_detect_line(ma, df, mdm, nr)
                if count == 12:
                    matching_configs.append((ma, df, mdm, nr))

print(f"Found {len(matching_configs)} matching configurations.")
if matching_configs:
    print("Sample config (min_area, delta_floor, max_diff_min, nms_radius):", matching_configs[0])
