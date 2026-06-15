import cv2
import glob
import numpy as np
from skimage.metrics import structural_similarity as ssim

def _ssim(a, b):
    a = cv2.cvtColor(cv2.resize(a, (256, 256)), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(b, (256, 256)), cv2.COLOR_BGR2GRAY)
    score, _ = ssim(a, b, full=True)
    return score

for f in sorted(glob.glob('validation_dataset/puzzle_*.jpg')):
    img = cv2.imread(f)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    prof_y = gray.mean(axis=1)
    lo, hi = int(h * 0.2), int(h * 0.8)
    smoothed = np.convolve(prof_y, np.ones(10)/10, mode='same')
    best_score = -1
    best_sep = -1
    for sep in range(lo, hi):
        if smoothed[sep] > smoothed[sep-2] and smoothed[sep] > smoothed[sep+2]:
            a, b = img[:sep], img[sep:]
            min_h = min(a.shape[0], b.shape[0])
            score = _ssim(a[-min_h:], b[:min_h])
            if score > best_score:
                best_score = score
                best_sep = sep
                
    prof_x = gray.mean(axis=0)
    lo, hi = int(w * 0.2), int(w * 0.8)
    smoothed_x = np.convolve(prof_x, np.ones(10)/10, mode='same')
    best_score_x = -1
    best_sep_x = -1
    for sep in range(lo, hi):
        if smoothed_x[sep] > smoothed_x[sep-2] and smoothed_x[sep] > smoothed_x[sep+2]:
            a, b = img[:, :sep], img[:, sep:]
            min_w = min(a.shape[1], b.shape[1])
            score = _ssim(a[:, -min_w:], b[:, :min_w])
            if score > best_score_x:
                best_score_x = score
                best_sep_x = sep
                
    print(f"=== {f} ===")
    print(f"H-cut at {best_sep} (SSIM {best_score:.3f}) | V-cut at {best_sep_x} (SSIM {best_score_x:.3f})")

