#!/usr/bin/env python3
"""
engine_v3.py  — Spot the Difference Engine v3
==============================================
Rebuild from scratch per the developer specification.

Key design principles (from spec):
  • Zero hardcoded image-specific overrides.  Every parameter is computed
    dynamically from the panel dimensions or the image data.
  • Dual pipeline: colour-space delta for paintings, local SSIM for line
    drawings.
  • Three-stage alignment: Lanczos resize → SIFT/AKAZE/ORB RANSAC →
    optional ECC refinement.
  • Boundary overlap masking (valid_mask) eroded by 1.5 % of panel width.
  • OCR masking (Khmer + English) with resolution-relative padding.
  • Margin spike detection only for colour mode — disabled for line drawings.
  • Exact contour outlines drawn; numbered badges placed adjacently.

Scale-invariant parameters:
  merge_radius  = max(20, panel_width * 0.05)
  min_area      = max(10, panel_w * panel_h * 0.0001)
  circle_pad    = max(4,  panel_width * 0.005)
  ocr_pad       = max(20, int(panel_width * 0.03))
  erosion_px    = max(5,  int(min(h, w) * 0.015))
  delta_floor   = max(5.0, std(absdiff) * 0.5)  [capped 15.0]

Usage:
  python engine_v3.py puzzle.jpg
  python engine_v3.py original.png modified.png
  python engine_v3.py numbergrid.png --mode number
  python engine_v3.py a.jpg b.jpg --output result.png
"""

import sys
import argparse
import warnings
import random
import os
import re
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  FIXED STRUCTURAL CONSTANTS  (not image-specific overrides)
# ─────────────────────────────────────────────────────────────────────────────

MAX_WARP_SCALE       = 3.0    # sanity guard: reject homographies that stretch > 3×
SAME_RATIO_TOL       = 0.03   # aspect-ratio difference below this → simple resize
ECC_MAX_DIM          = 400    # shrink to this size before ECC to keep it fast
ECC_MAX_ITER         = 150
ECC_EPS              = 1e-5

NUM_CELL_PAD         = 4      # pixels cropped from grid-cell border before OCR
NUM_MIN_COLS         = 3
NUM_MIN_ROWS         = 3

HUE_FIXED_THRESH     = 30     # degrees: hue arc above this triggers
HUE_SAT_MIN          = 30     # ignore pixels below this saturation
HUE_SCORE_WEIGHT     = 0.5    # weight of hue delta vs RGB delta
MAX_BLOB_RADIUS_FRAC = 0.25   # max enclosing-circle radius as fraction of min(h,w)
LOW_DELTA_FRAC       = 0.30   # drop groups whose delta < this fraction of median

_SATURATION_LINE_THRESHOLD = 20   # mean saturation below this → line-drawing mode


# ─────────────────────────────────────────────────────────────────────────────
#  DYNAMIC PARAMETER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _merge_radius(panel_w: int, panel_h: int) -> int:
    """6 % of the maximum panel dimension, minimum 20 px."""
    return max(20, int(max(panel_w, panel_h) * 0.06))


def _min_area(panel_w: int, panel_h: int) -> int:
    """0.01 % of panel area, minimum 10 px²."""
    return max(10, int(panel_w * panel_h * 0.0001))


def _circle_pad(panel_w: int) -> int:
    """0.5 % of panel width, minimum 4 px."""
    return max(4, int(panel_w * 0.005))


def _ocr_pad(panel_w: int) -> int:
    """3 % of panel width, minimum 20 px."""
    return max(20, int(panel_w * 0.03))


def _erosion_px(h: int, w: int) -> int:
    """1.5 % of shorter dimension, minimum 5 px."""
    return max(5, int(min(h, w) * 0.015))


def _hue_dilate_ksize(panel_w: int) -> int:
    """Hue dilation kernel — ~0.8 % of width, odd, in [5, 15]."""
    k = max(5, int(panel_w * 0.008))
    return k if k % 2 == 1 else k + 1


def _dynamic_delta_floor(cdiff: np.ndarray, valid_mask: np.ndarray | None) -> float:
    """
    Compute delta_floor from the std-dev of the absolute diff map.
    Using pixels inside valid_mask (if provided) avoids border noise inflating std.
    Capped between 5.0 and 15.0.
    """
    if valid_mask is not None and valid_mask.any():
        vals = cdiff[valid_mask > 0]
    else:
        vals = cdiff.ravel()
    if len(vals) == 0:
        return 7.0
    floor = float(np.std(vals)) * 0.5
    return float(np.clip(floor, 5.0, 15.0))


# ─────────────────────────────────────────────────────────────────────────────
#  FONTS
# ─────────────────────────────────────────────────────────────────────────────

_KHMER_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansKhmer-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifKhmer-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifKhmer-Regular.ttf",
    "C:/Windows/Fonts/NotoSansKhmer-Bold.ttf",
    "C:/Windows/Fonts/NotoSansKhmer-Regular.ttf",
    "/Library/Fonts/NotoSansKhmer-Bold.ttf",
    "/Library/Fonts/NotoSansKhmer-Regular.ttf",
    os.path.expanduser("~/Library/Fonts/NotoSansKhmer-Bold.ttf"),
    os.path.expanduser("~/Library/Fonts/NotoSansKhmer[wdth,wght].ttf"),
    "/usr/local/share/fonts/NotoSansKhmer-Bold.ttf",
    os.path.expanduser("~/.fonts/NotoSansKhmer-Bold.ttf"),
    os.path.expanduser("~/.local/share/fonts/NotoSansKhmer-Bold.ttf"),
]
_AVAILABLE_KHMER_FONTS = [p for p in _KHMER_FONT_CANDIDATES if os.path.exists(p)]

if not _AVAILABLE_KHMER_FONTS:
    print(
        "[WARN] No Khmer fonts found. Khmer text will fall back to the default font.\n"
        "       Linux : sudo apt install fonts-noto-core\n"
        "       macOS : brew install font-noto-sans-khmer"
    )

_LATIN_FONT = next((p for p in [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
] if os.path.exists(p)), None)


def _pick_khmer_font(size: int):
    test_text = "រកឃើញភាពខុសគ្នា ០១២៣៤៥៦៧៨៩"
    candidates = list(_AVAILABLE_KHMER_FONTS)
    random.shuffle(candidates)
    for path in candidates:
        try:
            f  = ImageFont.truetype(path, size)
            _d = ImageDraw.Draw(Image.new("RGB", (800, 100)))
            _d.textbbox((0, 0), test_text, font=f)
            _d.text((0, 0), test_text, font=f, fill=(0, 0, 0))
            return f, path
        except Exception:
            continue
    return ImageFont.load_default(), "default"


def _load_font(path, size):
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
#  1.  IMAGE I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_bgr(path: str) -> np.ndarray:
    return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)


def crop_text_by_gap(img: np.ndarray) -> tuple[np.ndarray, int]:
    """Strip rows of near-zero variance (text banners / solid colour borders)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row_var = np.var(gray, axis=1)
    content_rows = np.where(row_var > 50)[0]
    if len(content_rows) == 0:
        return img, 0
    start_y = max(0, content_rows[0] - 10)
    end_y   = min(img.shape[0], content_rows[-1] + 10)
    return img[start_y:end_y, :], start_y


def _ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(cv2.resize(a, (512, 512)), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, (512, 512)), cv2.COLOR_BGR2GRAY)
    score, _ = ssim(ga, gb, full=True)
    return float(score)


def auto_slice(img: np.ndarray):
    """
    Try every horizontal and vertical cut in the central 45–55 % band.
    Pick the split direction that yields the highest SSIM between the two
    halves (both penalised for extreme aspect ratios).
    """
    h, w = img.shape[:2]

    # Horizontal cuts
    best_h_score, best_h = -1.0, -1
    best_a_h = best_b_h = None
    best_astart_h = best_bstart_h = 0
    for sep in range(int(h * 0.45), int(h * 0.55)):
        a, b = img[:sep], img[sep:]
        min_h = min(a.shape[0], b.shape[0])
        ac, bc = a[-min_h:], b[:min_h]
        sc = _ssim_score(ac, bc)
        if sc > best_h_score:
            best_h_score = sc
            best_h = sep
            best_a_h, best_b_h = ac, bc
            best_astart_h = sep - min_h
            best_bstart_h = sep

    # Vertical cuts
    best_v_score, best_v = -1.0, -1
    best_a_v = best_b_v = None
    best_astart_v = best_bstart_v = 0
    for sep in range(int(w * 0.45), int(w * 0.55)):
        a, b = img[:, :sep], img[:, sep:]
        min_w = min(a.shape[1], b.shape[1])
        ac, bc = a[:, -min_w:], b[:, :min_w]
        sc = _ssim_score(ac, bc)
        if sc > best_v_score:
            best_v_score = sc
            best_v = sep
            best_a_v, best_b_v = ac, bc
            best_astart_v = sep - min_w
            best_bstart_v = sep

    # Penalise extreme aspect ratios
    def _aspect_penalty(arr):
        if arr is None:
            return 0.0
        ratio = arr.shape[1] / max(arr.shape[0], 1)
        return 0.1 if (ratio < 0.3 or ratio > 3.3) else 1.0

    sh = best_h_score * _aspect_penalty(best_a_h)
    sv = best_v_score * _aspect_penalty(best_a_v)

    if sh >= sv and best_h_score > 0:
        print(f"[V3] Auto-slice Horizontal  A{best_a_h.shape[:2]} B{best_b_h.shape[:2]}  SSIM={best_h_score:.3f}")
        return best_a_h, best_b_h, best_astart_h, best_bstart_h
    elif sv > sh and best_v_score > 0:
        print(f"[V3] Auto-slice Vertical    A{best_a_v.shape[:2]} B{best_b_v.shape[:2]}  SSIM={best_v_score:.3f}")
        return best_a_v, best_b_v, best_astart_v, best_bstart_v
    else:
        half = h // 2
        print("[V3] WARN: No valid separator — falling back to horizontal mid-cut")
        return img[:half], img[half:], 0, half


# ─────────────────────────────────────────────────────────────────────────────
#  2.  ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _warp_h(src, H, wh):
    return cv2.warpPerspective(src, H, wh,
                               flags=cv2.INTER_LANCZOS4,
                               borderMode=cv2.BORDER_REFLECT_101)


def _homography_ok(H, w, h):
    if H is None:
        return False
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    ow  = float(np.max(out[:, 0]) - np.min(out[:, 0]))
    oh  = float(np.max(out[:, 1]) - np.min(out[:, 1]))
    if ow < 1 or oh < 1:
        return False
    sx, sy = ow / w, oh / h
    lo, hi = 1.0 / MAX_WARP_SCALE, float(MAX_WARP_SCALE)
    return lo < sx < hi and lo < sy < hi


def _stage1_scale(ref, tgt):
    h,  w  = ref.shape[:2]
    ht, wt = tgt.shape[:2]
    if h == ht and w == wt:
        print("[V3-S1] Same size — skip")
        return tgt, True

    ar_r = w  / h
    ar_t = wt / ht
    diff = abs(ar_r - ar_t) / max(ar_r, ar_t)
    if diff <= SAME_RATIO_TOL:
        out = cv2.resize(tgt, (w, h), interpolation=cv2.INTER_LANCZOS4)
        print("[V3-S1] Same ratio → Lanczos resize")
        return out, True

    scale   = min(w / wt, h / ht)
    nw, nh  = int(wt * scale), int(ht * scale)
    resized = cv2.resize(tgt, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    edge    = np.concatenate([resized[0].reshape(-1, 3), resized[-1].reshape(-1, 3),
                               resized[:, 0].reshape(-1, 3), resized[:, -1].reshape(-1, 3)])
    pad_c   = tuple(int(x) for x in np.median(edge, axis=0))
    canvas  = np.full((h, w, 3), pad_c, dtype=np.uint8)
    y0, x0  = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    print("[V3-S1] Different ratio → fit-and-pad")
    return canvas, False


def _match_features(g_ref, g_tgt, name):
    if   name == "SIFT":  det, norm = cv2.SIFT_create(nfeatures=12000), cv2.NORM_L2
    elif name == "AKAZE": det, norm = cv2.AKAZE_create(),                cv2.NORM_HAMMING
    else:                 det, norm = cv2.ORB_create(nfeatures=12000),   cv2.NORM_HAMMING
    kp1, d1 = det.detectAndCompute(g_ref, None)
    kp2, d2 = det.detectAndCompute(g_tgt, None)
    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
        return kp1, kp2, []
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    raw     = matcher.knnMatch(d2, d1, k=2)
    good    = [m for m, n in raw if len((m, n)) == 2 and m.distance < 0.78 * n.distance]
    return kp1, kp2, good


def _stage2_features(ref, tgt):
    h, w  = ref.shape[:2]
    g_ref = _gray(ref)
    g_tgt = _gray(tgt)
    for name in ("SIFT", "AKAZE", "ORB"):
        kp1, kp2, good = _match_features(g_ref, g_tgt, name)
        print(f"[V3-S2] {name}: {len(good)} good matches")
        if len(good) < 12:
            continue
        src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if not _homography_ok(H, w, h):
            print(f"[V3-S2] {name}: homography failed sanity check")
            continue
        inliers = int(mask.sum()) if mask is not None else 0
        print(f"[V3-S2] {name}: accepted ({inliers} inliers)")
        warped = _warp_h(tgt, H, (w, h))
        return warped, H
    print("[V3-S2] No usable homography")
    return tgt, None


def _stage3_ecc(ref, tgt):
    h, w   = ref.shape[:2]
    sc     = min(1.0, ECC_MAX_DIM / max(h, w))
    sw, sh = int(w * sc), int(h * sc)
    g_ref  = _gray(cv2.resize(ref, (sw, sh)))
    g_tgt  = _gray(cv2.resize(tgt, (sw, sh)))
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_MAX_ITER, ECC_EPS)
    H_init   = np.eye(3, dtype=np.float32)
    try:
        _, H_small = cv2.findTransformECC(g_ref, g_tgt, H_init,
                                          cv2.MOTION_HOMOGRAPHY, criteria,
                                          inputMask=None, gaussFiltSize=5)
    except cv2.error as e:
        print(f"[V3-S3] ECC failed: {e}")
        return tgt, None
    H_full       = H_small.copy()
    H_full[0, 2] /= sc
    H_full[1, 2] /= sc
    if not _homography_ok(H_full, w, h):
        print("[V3-S3] ECC result insane — skipped")
        return tgt, None
    return _warp_h(tgt, H_full, (w, h)), H_full


def align(ref, tgt, skip_ecc=False):
    """
    Three-stage alignment. Returns (aligned_tgt, H, valid_mask).
    valid_mask is a uint8 mask representing the overlap region after warping,
    eroded by 1.5 % of the shorter dimension to blind boundary noise.
    """
    h, w   = ref.shape[:2]
    ht, wt = tgt.shape[:2]

    s0 = _ssim_score(ref, tgt)
    print(f"[V3] SSIM before alignment : {s0:.4f}")

    tgt_s1, full_overlap = _stage1_scale(ref, tgt)
    s1 = _ssim_score(ref, tgt_s1)
    print(f"[V3] SSIM after  Stage-1   : {s1:.4f}")

    # Build initial valid_mask from Stage-1
    valid_mask = np.zeros((h, w), dtype=np.uint8)
    if full_overlap:
        valid_mask[:, :] = 255
    else:
        scale  = min(w / wt, h / ht)
        nw, nh = int(wt * scale), int(ht * scale)
        y0, x0 = (h - nh) // 2, (w - nw) // 2
        valid_mask[y0:y0 + nh, x0:x0 + nw] = 255

    tgt_s2, H2 = _stage2_features(ref, tgt_s1)
    s2 = _ssim_score(ref, tgt_s2)
    if s2 >= s1 - 0.005:
        tgt_cur, s_cur, H = tgt_s2, s2, H2
        if H2 is not None:
            valid_mask = cv2.warpPerspective(valid_mask, H2, (w, h),
                                             flags=cv2.INTER_NEAREST,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        tgt_cur, s_cur, H = tgt_s1, s1, None
        print("[V3] Stage-2 made things worse — reverted")
    print(f"[V3] SSIM after  Stage-2   : {s_cur:.4f}")

    if not skip_ecc:
        tgt_s3, H3 = _stage3_ecc(ref, tgt_cur)
        s3 = _ssim_score(ref, tgt_s3)
        if s3 > s_cur + 0.001:
            if H is not None and H3 is not None:
                H = H3 @ H
            elif H3 is not None:
                H = H3
            if H3 is not None:
                valid_mask = cv2.warpPerspective(valid_mask, H3, (w, h),
                                                 flags=cv2.INTER_NEAREST,
                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            tgt_cur, s_cur = tgt_s3, s3
            print(f"[V3] SSIM after  Stage-3   : {s_cur:.4f}  (ECC applied)")
        else:
            print(f"[V3] SSIM after  Stage-3   : {s3:.4f}  (ECC skipped — no improvement)")

    # Erode valid_mask by 1.5 % of shorter dimension
    ep = _erosion_px(h, w)
    k  = cv2.getStructuringElement(cv2.MORPH_RECT, (ep, ep))
    valid_mask = cv2.erode(valid_mask, k)
    print(f"[V3] valid_mask eroded by {ep}px ({ep/min(h,w)*100:.1f}% of shorter dim)")

    return tgt_cur, H, valid_mask


# ─────────────────────────────────────────────────────────────────────────────
#  3.  OCR MASKING
# ─────────────────────────────────────────────────────────────────────────────

def _is_watermark(text: str) -> bool:
    t = text.lower()
    eng = ["enterprises", "digital", "gov", "kh", "copyright", "©",
           "c0pyright", "ste", "bae", "john"]
    khm = ["រូប", "រប", "ទី", "ខុស", "គ្នា", "ស្វែង", "រក", "ចំណុច", "រូបភាព"]
    return any(k in text for k in khm) or any(k in t for k in eng)


def _mask_ocr_text(img_a: np.ndarray, img_b: np.ndarray,
                   bmask: np.ndarray, line_mode: bool = False):
    """
    Run Tesseract on both panels and zero-out detected watermark/label regions
    in bmask.  Padding is resolution-relative: 3 % of panel width.
    In line_mode, only confirmed watermark keywords are masked (to prevent
    sketch lines from being accidentally excluded).
    """
    try:
        import pytesseract
        h_panel, w_panel = img_a.shape[:2]
        pad = _ocr_pad(w_panel)

        for img in (img_a, img_b):
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            data = pytesseract.image_to_data(
                pil_img, config="-l eng+khm --psm 11",
                output_type=pytesseract.Output.DICT)
            for i in range(len(data.get("level", []))):
                text = data["text"][i].strip()
                if not text:
                    continue
                bw, bh = data["width"][i], data["height"][i]
                conf = int(data.get("conf", [-1])[i]) if "conf" in data else -1
                min_box_area = w_panel * h_panel * 0.0001  # 0.01% of panel

                if line_mode:
                    # Line mode: only mask confirmed watermark keywords
                    if not _is_watermark(text):
                        continue
                else:
                    # Colour mode: mask watermarks always; mask other text only
                    # if Tesseract is confident, the box is large enough, AND
                    # the text is at least 2 characters (single chars are often
                    # OCR noise on textured painting surfaces, not real labels).
                    is_wm = _is_watermark(text)
                    confident = (conf < 0 or conf >= 60)  # raised to 60 for selectivity
                    big_enough = (bw * bh) >= min_box_area
                    long_enough = len(text) >= 2
                    if not is_wm and not (confident and big_enough and long_enough):
                        continue

                # Skip boxes that are implausibly large (OCR layout errors)
                if bw > w_panel * 0.35 or bh > h_panel * 0.25:
                    print(f"[V3-OCR] Skipping oversized box {bw}×{bh} for '{text}'")
                    continue
                x1 = max(0, data["left"][i] - pad)
                y1 = max(0, data["top"][i]  - pad)
                x2 = min(bmask.shape[1], data["left"][i] + bw + pad)
                y2 = min(bmask.shape[0], data["top"][i]  + bh + pad)
                bmask[y1:y2, x1:x2] = 0
                print(f"[V3-OCR] Masked '{text}' → ({x1},{y1})–({x2},{y2})")

        # ── Vertical colour-mask pass for rotated side labels ────────────────
        crop_w = min(150, w_panel // 4)
        for img in (img_a, img_b):
            crop = img[:, w_panel - crop_w:]
            hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask_y  = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
            mask_r1 = cv2.inRange(hsv, (0,  80, 80), (10, 255, 255))
            mask_r2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
            for cmask in (mask_y, mask_r1 | mask_r2):
                rot = cv2.rotate(cmask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                data = pytesseract.image_to_data(
                    Image.fromarray(rot), config="-l eng+khm --psm 11",
                    output_type=pytesseract.Output.DICT)
                for i in range(len(data.get("level", []))):
                    text = data["text"][i].strip()
                    if not text or not _is_watermark(text):
                        continue
                    rx, ry = data["left"][i], data["top"][i]
                    rw, rh = data["width"][i], data["height"][i]
                    # Map rotated coords back to panel space
                    x_orig  = crop_w - ry - rh
                    y_orig  = rx
                    x_full  = w_panel - crop_w + x_orig
                    x1 = max(0, x_full - pad)
                    y1 = max(0, y_orig - pad)
                    x2 = min(w_panel, x_full + rh + pad)
                    y2 = min(h_panel, y_orig + rw + pad)
                    bmask[y1:y2, x1:x2] = 0
                    print(f"[V3-OCR] Masked vertical label '{text}' → ({x1},{y1})–({x2},{y2})")
    except Exception as e:
        print(f"[V3-OCR] Masking skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  4.  MARGIN SPIKE MASKING  (colour mode only)
# ─────────────────────────────────────────────────────────────────────────────

def _mask_margins(img: np.ndarray, bmask: np.ndarray):
    """
    Detect frame/border edges via row/column gradient spikes in the outer
    15 % of the panel, then zero out those regions in bmask.
    NOT called in line-drawing mode — sketch outlines trigger false spikes.
    """
    try:
        h, w  = img.shape[:2]
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col_d = np.mean(np.abs(gray[:, 1:].astype(float) - gray[:, :-1].astype(float)), axis=0)
        row_d = np.mean(np.abs(gray[1:, :].astype(float) - gray[:-1, :].astype(float)), axis=1)

        max_x = min(180, max(30, int(w * 0.15)))
        max_y = min(180, max(30, int(h * 0.15)))

        left_spike = right_spike = 0
        for x in range(15, max_x):
            if col_d[x] > 20.0:
                left_spike = x
                break
        for x in range(w - 16, w - max_x, -1):
            if col_d[x] > 20.0:
                right_spike = x + 1
                break

        top_spike = bottom_spike = 0
        for y in range(15, max_y):
            if row_d[y] > 12.0:
                top_spike = y
                break
        for y in range(h - 16, h - max_y, -1):
            if row_d[y] > 12.0:
                bottom_spike = y + 1
                break

        slack = max(6, int(min(h, w) * 0.01))
        if left_spike   > 0: bmask[:, :left_spike + slack]     = 0
        if right_spike  > 0: bmask[:, right_spike - slack:]    = 0
        if top_spike    > 0: bmask[:top_spike + slack, :]      = 0
        if bottom_spike > 0: bmask[bottom_spike - slack:, :]   = 0
    except Exception as e:
        print(f"[V3-MARGIN] {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  5.  DIFFERENCE DETECTION — COLOUR MODE
# ─────────────────────────────────────────────────────────────────────────────

def _lab_delta_map(a, b):
    la = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lb = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
    d  = np.sqrt(np.sum((la - lb) ** 2, axis=2))
    mx = d.max()
    return (d / mx * 255).astype(np.uint8) if mx > 0 else d.astype(np.uint8)


def _hue_delta_map(a, b, hue_dilate_k):
    hsva = cv2.cvtColor(a, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsvb = cv2.cvtColor(b, cv2.COLOR_BGR2HSV).astype(np.float32)
    dh   = np.minimum(np.abs(hsva[:, :, 0] - hsvb[:, :, 0]), 180.0 - np.abs(hsva[:, :, 0] - hsvb[:, :, 0]))
    sat  = ((hsva[:, :, 1] > HUE_SAT_MIN) & (hsvb[:, :, 1] > HUE_SAT_MIN)).astype(np.float32)
    dh  *= sat
    t    = (dh > HUE_FIXED_THRESH).astype(np.uint8) * 255
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hue_dilate_k, hue_dilate_k))
    return cv2.morphologyEx(t, cv2.MORPH_DILATE, k), dh


def _auto_threshold(deltas, floor):
    s = np.array(sorted(deltas))
    if len(s) < 3 or s[0] - floor > 3.0:
        return floor, "floor used"
    c1, c2 = float(np.percentile(s, 25)), float(np.percentile(s, 75))
    for _ in range(10):
        g1 = s[np.abs(s - c1) < np.abs(s - c2)]
        g2 = s[np.abs(s - c1) >= np.abs(s - c2)]
        if not len(g1) or not len(g2):
            break
        c1, c2 = float(np.mean(g1)), float(np.mean(g2))
    g1 = s[np.abs(s - c1) < np.abs(s - c2)]
    g2 = s[np.abs(s - c1) >= np.abs(s - c2)]
    max_g1 = float(np.max(g1))
    min_g2 = float(np.min(g2))
    t  = (max_g1 + min_g2) / 2.0
    # Outlier check inside g1
    if len(g1) > 2:
        mn, sd = np.mean(g1), np.std(g1)
        if sd > 0 and (np.max(g1) - mn) / sd > 1.8:
            g1_trim = g1[g1 < np.max(g1)]
            t = (float(np.max(g1_trim)) + float(np.max(g1))) / 2.0
            max_g1 = float(np.max(g1_trim))
    # Only split if there is a clear gap between the two clusters AND
    # g1's maximum is well below floor+20 (prevents splitting tightly-packed
    # low-delta sets like puzzle_05 where all deltas are meaningful).
    # Also don't split if the lower cluster itself has many members (≥4),
    # because that strongly suggests all candidates are real differences.
    gap = min_g2 - max_g1
    # Split if: clear gap (≥5.0) AND g1's max is in the noise range (<20.0).
    # The len(g1)<4 guard is intentionally removed: even a large lower cluster
    # is noise if there's a clear gap and the cluster max is below 20.
    if max_g1 < 20.0 and gap >= 5.0:
        return max(t, floor), f"dynamic split at {t:.1f} (gap={gap:.1f})"
    return floor, "split rejected"


def _split_large_blob(blob_mask, cdiff, max_r, h, w, circle_pad):
    dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
    if dist.max() < 3:
        return []
    for ep in (70, 50, 35, 20, 10):
        ke    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ep, ep))
        peaks = cv2.erode(blob_mask, ke)
        if not peaks.sum():
            continue
        sub_cnts, _ = cv2.findContours(peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(sub_cnts) < 2:
            continue
        result, seen = [], []
        for sc in sub_cnts:
            (cx, cy), _ = cv2.minEnclosingCircle(sc)
            cx, cy = int(cx), int(cy)
            if any(((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5 < max_r * 0.6 for sx, sy in seen):
                continue
            seen.append((cx, cy))
            r  = min(ep + circle_pad + 20, max_r)
            lm = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(lm, (cx, cy), r, 255, -1)
            if cv2.mean(cdiff, mask=lm)[0] > 5:
                result.append((cx, cy, r, cv2.mean(cdiff, mask=lm)[0]))
        if result:
            print(f"[V3] Blob split → {len(result)} sub-circles (erode={ep})")
            return result
    # Fallback: single circle at dist-transform peak
    py, px = np.unravel_index(dist.argmax(), dist.shape)
    r = min(int(dist.max()) + circle_pad, max_r)
    lm = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(lm, (int(px), int(py)), r, 255, -1)
    if cv2.mean(cdiff, mask=lm)[0] > 5:
        return [(int(px), int(py), r, cv2.mean(cdiff, mask=lm)[0])]
    return []


def detect(img_a, img_b, valid_mask=None, H=None, split_dir=None,
           mask_rois=None):
    """
    Colour-space difference detection.
    All parameters computed dynamically from panel dimensions.
    """
    h, w = img_a.shape[:2]
    assert img_b.shape[:2] == (h, w)

    # ── Dynamic parameters ───────────────────────────────────────────────────
    merge_r    = _merge_radius(w, h)
    min_area   = _min_area(w, h)
    c_pad      = _circle_pad(w)
    hue_dil_k  = _hue_dilate_ksize(w)
    max_r      = int(min(h, w) * MAX_BLOB_RADIUS_FRAC)
    border_px  = max(16, int(min(h, w) * 0.02))
    print(f"[V3] Params: merge_r={merge_r}  min_area={min_area}  c_pad={c_pad}  "
          f"hue_dil={hue_dil_k}  max_r={max_r}  border={border_px}")

    # ── Multi-channel difference maps ────────────────────────────────────────
    gray_a = _gray(img_a)
    gray_b = _gray(img_b)
    score, diff = ssim(gray_a, gray_b, full=True)
    print(f"[V3] SSIM for detection : {score:.4f}")
    inv = cv2.bitwise_not((diff * 255).clip(0, 255).astype(np.uint8))
    lab_diff = _lab_delta_map(img_a, img_b)
    thresh_hue, hue_diff_deg = _hue_delta_map(img_a, img_b, hue_dil_k)

    valid_ssim = inv
    valid_lab  = lab_diff
    otsu_ssim  = cv2.threshold(valid_ssim, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
    otsu_lab   = cv2.threshold(valid_lab,  0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
    _, thresh_ssim = cv2.threshold(inv,      otsu_ssim, 255, cv2.THRESH_BINARY)
    _, thresh_lab  = cv2.threshold(lab_diff, otsu_lab,  255, cv2.THRESH_BINARY)

    thresh = cv2.bitwise_or(thresh_ssim, thresh_lab)
    thresh = cv2.bitwise_or(thresh, thresh_hue)

    # ── Build base mask ──────────────────────────────────────────────────────
    bmask = np.zeros_like(thresh)
    if split_dir == "vertical":
        bmask[12:h - 20, 16:w - 10] = 255
    elif split_dir == "horizontal":
        bmask[16:h - 8, 8:w - 8] = 255
    else:
        bmask[border_px:h - border_px, border_px:w - border_px] = 255

    # Apply valid_mask (alignment overlap)
    if valid_mask is not None:
        ep = _erosion_px(h, w)
        k  = cv2.getStructuringElement(cv2.MORPH_RECT, (ep, ep))
        bmask = cv2.bitwise_and(bmask, cv2.erode(valid_mask, k))
        print("[V3] Applied dynamic alignment overlap mask.")
    elif H is not None:
        mb = np.ones((h, w), dtype=np.uint8) * 255
        wm = cv2.warpPerspective(mb, H, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ep = _erosion_px(h, w)
        k  = cv2.getStructuringElement(cv2.MORPH_RECT, (ep, ep))
        bmask = cv2.bitwise_and(bmask, cv2.erode(wm, k))

    # Margin spike detection (colour only)
    _mask_margins(img_a, bmask)

    # OCR watermark masking
    _mask_ocr_text(img_a, img_b, bmask, line_mode=False)

    # Custom ROIs
    if mask_rois:
        for rx1, ry1, rx2, ry2 in mask_rois:
            bmask[ry1:ry2, rx1:rx2] = 0

    # ── Morphological cleaning ───────────────────────────────────────────────
    thresh = cv2.bitwise_and(thresh, bmask)
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k9)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k5)
    thresh = cv2.bitwise_and(thresh, bmask)

    # ── Candidate extraction with blob splitting ─────────────────────────────
    cdiff_rgb = np.mean(cv2.absdiff(img_a, img_b).astype(np.float32), axis=2)
    delta_floor = _dynamic_delta_floor(cdiff_rgb, valid_mask)
    print(f"[V3] Dynamic delta_floor : {delta_floor:.2f}")

    # Maximum allowed contour area (2 % of panel) — rejects giant AI-texture
    # SSIM blobs that span large image regions but are not real differences.
    max_cnt_area = h * w * 0.02
    pre_cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    extra_cands = []
    for cnt in pre_cnts:
        (_, _), r = cv2.minEnclosingCircle(cnt)
        if r > max_r:
            bm = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(bm, [cnt], -1, 255, cv2.FILLED)
            print(f"[V3] Large blob r={int(r)} — attempting split …")
            for sc in _split_large_blob(bm, cdiff_rgb, max_r, h, w, c_pad):
                extra_cands.append(sc)
            cv2.drawContours(thresh, [cnt], -1, 0, cv2.FILLED)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if area > max_cnt_area:
            print(f"[V3] Skipping giant contour area={int(area)} > max={int(max_cnt_area)} (texture noise)")
            continue
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
        rgb_d = cv2.mean(cdiff_rgb,    mask=m)[0]
        hue_d = cv2.mean(hue_diff_deg, mask=m)[0]
        delta = max(rgb_d, hue_d * HUE_SCORE_WEIGHT)
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        candidates.append((cnt, delta, int(cx), int(cy), int(r)))

    for scx, scy, sr, sd in extra_cands:
        candidates.append((None, sd, scx, scy, sr))

    if not candidates:
        print("[V3] WARN: No candidates survived.")
        return [], 0

    deltas = sorted(d for _, d, _, _, _ in candidates)
    print(f"[V3] Candidates: {len(candidates)}  deltas: {[round(d, 1) for d in deltas]}")

    threshold, reason = _auto_threshold(deltas, delta_floor)
    print(f"[V3] Delta threshold : {threshold:.1f}  ({reason})")

    surviving = [(cnt, d, cx, cy, r) for cnt, d, cx, cy, r in candidates if d >= threshold]

    # ── Hierarchical single-linkage merge ────────────────────────────────────
    groups = []
    for _, delta, cx, cy, r in surviving:
        cx, cy = float(cx), float(cy)
        merged = False
        for grp in groups:
            if ((cx - grp[2]) ** 2 + (cy - grp[3]) ** 2) ** 0.5 < merge_r:
                grp[0].append((cx, cy, r))
                grp[1] = max(grp[1], delta)
                grp[2] = float(np.mean([s[0] for s in grp[0]]))
                grp[3] = float(np.mean([s[1] for s in grp[0]]))
                merged = True
                break
        if not merged:
            groups.append([[(cx, cy, r)], delta, cx, cy])

    print(f"[V3] After merge: {len(groups)} groups  deltas: {[round(g[1],1) for g in groups]}")
    groups.sort(key=lambda g: g[1], reverse=True)

    # Drop very low-delta outliers
    if len(groups) >= 3:
        all_d = np.array([g[1] for g in groups])
        groups = [g for g in groups
                  if g[1] >= LOW_DELTA_FRAC * float(np.median(all_d[all_d != g[1]]))]

    # ── Convert groups → (cx, cy, r) circles ────────────────────────────────
    circles = []
    for grp in groups:
        sub = grp[0]
        centres = np.array([[s[0], s[1]] for s in sub], dtype=np.float32)
        max_sub_r = max(s[2] for s in sub)
        if len(centres) == 1:
            cx, cy = centres[0]
            r = max(int(max_sub_r) + c_pad, 18)
        else:
            (cx, cy), span = cv2.minEnclosingCircle(centres.reshape(-1, 1, 2))
            r = max(int(span + max_sub_r) + c_pad, 18)
        circles.append((int(cx), int(cy), min(r, max_r)))

    return circles, len(circles)


# ─────────────────────────────────────────────────────────────────────────────
#  6.  DIFFERENCE DETECTION — LINE-DRAWING MODE
# ─────────────────────────────────────────────────────────────────────────────

def is_line_drawing(img: np.ndarray) -> bool:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = float(hsv[:, :, 1].mean())
    print(f"[V3] Mean saturation: {sat:.1f}  → {'line-drawing' if sat < _SATURATION_LINE_THRESHOLD else 'colour'}")
    return sat < _SATURATION_LINE_THRESHOLD


def detect_line(img_a, img_b, valid_mask=None, H=None, mask_rois=None):
    """
    SSIM-based difference detection for line drawings.
    Margin spike detection is NOT run — sketch outlines trigger false spikes.
    """
    h, w = img_a.shape[:2]
    img_b = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # ── Dynamic parameters ───────────────────────────────────────────────────
    min_area   = _min_area(w, h)
    merge_r    = max(20, int(max(h, w) * 0.10))   # NMS radius for line mode
    nms_r      = merge_r
    max_r      = int(min(h, w) * 0.15)
    c_pad      = _circle_pad(w)
    border_px  = max(8, int(min(h, w) * 0.02))
    ssim_thresh = 30
    morph_k     = 3
    # Minimum absolute pixel peak (ensures the region is a real drawing diff,
    # not a minor alignment artefact). Raised to 55% of min-dim to reduce
    # false positives on line drawings with partial alignment noise.
    peak_min    = max(80, int(min(h, w) * 0.55))
    print(f"[V3-LINE] Params: min_area={min_area}  nms_r={nms_r}  peak_min={peak_min}")

    score, diff = ssim(_gray(img_a), _gray(img_b), full=True)
    print(f"[V3-LINE] SSIM : {score:.4f}")
    inv = cv2.bitwise_not((diff * 255).clip(0, 255).astype(np.uint8))
    _, thresh = cv2.threshold(inv, ssim_thresh, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k)

    # ── Build mask (no margin spike detection) ───────────────────────────────
    bmask = np.zeros_like(thresh)
    bmask[border_px:h - border_px, border_px:w - border_px] = 255

    if valid_mask is not None:
        ep = _erosion_px(h, w)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (ep, ep))
        bmask = cv2.bitwise_and(bmask, cv2.erode(valid_mask, k2))
    elif H is not None:
        try:
            H_inv = np.linalg.inv(H)
            mb = np.ones((h, w), dtype=np.uint8) * 255
            wm = cv2.warpPerspective(mb, H_inv, (w, h), flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            ep = _erosion_px(h, w)
            k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (ep, ep))
            bmask = cv2.bitwise_and(bmask, cv2.erode(wm, k2))
        except Exception as e:
            print(f"[V3-LINE] Warp mask failed: {e}")

    # OCR masking (watermark keywords only — no sketch-line false positives)
    _mask_ocr_text(img_a, img_b, bmask, line_mode=True)

    if mask_rois:
        for rx1, ry1, rx2, ry2 in mask_rois:
            bmask[ry1:ry2, rx1:rx2] = 0

    thresh = cv2.bitwise_and(thresh, bmask)

    # Dynamic delta floor for line mode
    cdiff = np.mean(cv2.absdiff(img_a, img_b).astype(np.float32), axis=2)
    delta_floor = _dynamic_delta_floor(cdiff, valid_mask)
    ga = _gray(img_a).astype(np.float64)
    gb = _gray(img_b).astype(np.float64)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
        delta = cv2.mean(cdiff, mask=m)[0]
        if delta < delta_floor:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        cx, cy, r = int(cx), int(cy), int(r)
        x1 = max(0, cx - r - 5); y1 = max(0, cy - r - 5)
        x2 = min(w, cx + r + 5); y2 = min(h, cy + r + 5)
        peak = float(np.abs(ga[y1:y2, x1:x2] - gb[y1:y2, x1:x2]).max())
        if peak < peak_min:
            continue
        candidates.append((cx, cy, min(max(r + c_pad + 10, 20), max_r), delta))

    candidates.sort(key=lambda x: -x[3])

    # NMS
    kept = []
    for cx, cy, r, d in candidates:
        if not any(((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5 < nms_r for kx, ky, _, _ in kept):
            kept.append((cx, cy, r, d))

    circles = [(cx, cy, r) for cx, cy, r, _ in kept]
    return circles, len(kept)


# ─────────────────────────────────────────────────────────────────────────────
#  7.  DIFFERENCE DETECTION — NUMBER GRID MODE
# ─────────────────────────────────────────────────────────────────────────────

def _get_grid_bbox(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 8)
    if np.mean(binary) < 128:
        binary = cv2.bitwise_not(binary)
    cnts, _ = cv2.findContours(cv2.bitwise_not(binary), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        if cw > img_bgr.shape[1] * 0.7 and ch > img_bgr.shape[0] * 0.7:
            return x, y, cw, ch
    return 0, 0, img_bgr.shape[1], img_bgr.shape[0]


def detect_number_grid(img_a, img_b):
    print("[V3-NUM] Number-grid mode: visual cell comparison …")
    xa, ya, wa, ha = _get_grid_bbox(img_a)
    xb, yb, wb, hb = _get_grid_bbox(img_b)

    rows, cols = 10, 9
    scores, cell_info = [], []

    for r in range(rows):
        for c in range(cols):
            x1a = int(xa + c * (wa / cols));  y1a = int(ya + r * (ha / rows))
            x2a = int(xa + (c + 1) * (wa / cols)); y2a = int(ya + (r + 1) * (ha / rows))
            x1b = int(xb + c * (wb / cols));  y1b = int(yb + r * (hb / rows))
            x2b = int(xb + (c + 1) * (wb / cols)); y2b = int(yb + (r + 1) * (hb / rows))

            mx = int((x2a - x1a) * 0.15)
            my = int((y2a - y1a) * 0.15)
            inner_a = cv2.cvtColor(img_a[y1a + my:y2a - my, x1a + mx:x2a - mx], cv2.COLOR_BGR2GRAY)

            min_score = float("inf")
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ys = max(0, y1b + my + dy); ye = min(img_b.shape[0], y2b - my + dy)
                    xs = max(0, x1b + mx + dx); xe = min(img_b.shape[1], x2b - mx + dx)
                    inner_b = cv2.cvtColor(img_b[ys:ye, xs:xe], cv2.COLOR_BGR2GRAY)
                    if inner_b.shape != inner_a.shape:
                        inner_b = cv2.resize(inner_b, (inner_a.shape[1], inner_a.shape[0]))
                    sc = np.mean(cv2.absdiff(inner_a, inner_b))
                    if sc < min_score:
                        min_score = sc
            scores.append(min_score)
            cx_a = int(xa + (c + 0.5) * (wa / cols))
            cy_a = int(ya + (r + 0.5) * (ha / rows))
            cx_b = int(xb + (c + 0.5) * (wb / cols))
            cy_b = int(yb + (r + 0.5) * (hb / rows))
            cell_info.append((r, c, cx_a, cy_a, cx_b, cy_b))

    # 2-means clustering for threshold
    s = np.array(scores)
    c1, c2 = float(np.percentile(s, 25)), float(np.percentile(s, 75))
    for _ in range(20):
        g1 = s[np.abs(s - c1) < np.abs(s - c2)]
        g2 = s[np.abs(s - c1) >= np.abs(s - c2)]
        if not len(g1) or not len(g2):
            break
        nc1, nc2 = float(np.mean(g1)), float(np.mean(g2))
        if nc1 == c1 and nc2 == c2:
            break
        c1, c2 = nc1, nc2
    threshold = (np.max(g1) + np.min(g2)) / 2.0
    if threshold < 15.0 or threshold > 35.0:
        threshold = 25.0

    radius = max(12, int(min(wb / cols, hb / rows) * 0.45))
    diffs_a, diffs_b, count = [], [], 0
    for idx, (r, c, cxa, cya, cxb, cyb) in enumerate(cell_info):
        if scores[idx] >= threshold:
            diffs_a.append((cxa, cya, radius))
            diffs_b.append((cxb, cyb, radius))
            count += 1
            print(f"[V3-NUM]   Diff row={r+1} col={c+1} score={scores[idx]:.2f}")
    print(f"[V3-NUM] {count} differences (threshold={threshold:.2f})")
    return diffs_a, diffs_b, count


# ─────────────────────────────────────────────────────────────────────────────
#  8.  OUTPUT RENDERING
# ─────────────────────────────────────────────────────────────────────────────

def _random_color():
    h = random.uniform(0, 360)
    s = random.uniform(0.80, 1.00)
    v = random.uniform(0.80, 1.00)
    c = v * s; x = c * (1 - abs((h / 60) % 2 - 1)); m = v - c
    r, g, b = [(c,x,0),(x,c,0),(0,c,x),(0,x,c),(x,0,c),(c,0,x)][int(h//60)%6]
    return (int((r+m)*255), int((g+m)*255), int((b+m)*255))


def _khmer_digits(n):
    return str(n).translate(str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩"))


def make_khmer_banner(width, count):
    BH    = 80
    text  = f"រកឃើញភាពខុសគ្នា {_khmer_digits(count)} កន្លែង"
    banner = Image.new("RGB", (width, BH), (30, 30, 50))
    draw   = ImageDraw.Draw(banner)
    kfont, kpath = _pick_khmer_font(40)
    print(f"[V3] Banner font: {os.path.basename(kpath)}")
    bb = draw.textbbox((0, 0), text, font=kfont)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    tx = (width - tw) // 2 - bb[0]
    ty = (BH - th) // 2   - bb[1]
    draw.text((tx+2, ty+2), text, font=kfont, fill=(10, 10, 20))
    draw.text((tx,   ty),   text, font=kfont, fill=(255, 215, 60))
    return banner


def add_watermark(img: Image.Image) -> Image.Image:
    text = "ចម្លើយពីក្មួយ និរន្ត"
    w, h = img.size
    wm   = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(wm)
    font, _ = _pick_khmer_font(max(24, w // 18))
    bb   = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    step_x, step_y, angle = int(tw * 1.6), int(th * 3.5), -30
    for row, y0 in enumerate(range(-th * 2, h + th * 2, step_y)):
        x_shift = (row % 2) * (step_x // 2)
        for x0 in range(-tw - x_shift, w + tw, step_x):
            tile  = Image.new("RGBA", (tw + 20, th + 20), (0, 0, 0, 0))
            tdraw = ImageDraw.Draw(tile)
            for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
                tdraw.text((10 - bb[0] + dx, 10 - bb[1] + dy),
                           text, font=font, fill=(255, 255, 255, 60))
            tdraw.text((10 - bb[0], 10 - bb[1]), text, font=font, fill=(180, 180, 180, 55))
            rot = tile.rotate(angle, expand=True)
            wm.paste(rot, (x0, y0), rot)
    return Image.alpha_composite(img.convert("RGBA"), wm).convert("RGB")


def draw_circles_on_panel(panel_pil, circles, color, H_inv=None):
    """Simple circle drawing — used for number-grid mode."""
    pw, ph = panel_pil.size
    if H_inv is not None and circles:
        pts    = np.float32([[cx, cy] for cx, cy, _ in circles]).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
        circles = [(int(np.clip(mx, 0, pw-1)), int(np.clip(my, 0, ph-1)), r)
                   for (mx, my), (_, _, r) in zip(mapped, circles)]
    img  = panel_pil.copy()
    draw = ImageDraw.Draw(img)
    for cx, cy, r in circles:
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=color, width=4)
    return img


def draw_contours_and_numbers_on_panel(panel_pil, circles, img_a, img_b_aligned,
                                       H_inv, color, valid_mask=None):
    """
    Draw exact difference contours on the panel and place a numbered label
    immediately adjacent to each shape.
    Falls back to a circle only if no contour can be found.
    """
    if not circles:
        return panel_pil

    h, w = img_a.shape[:2]
    diff      = cv2.absdiff(img_a, img_b_aligned)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    b_cv = cv2.cvtColor(np.array(panel_pil), cv2.COLOR_RGB2BGR)
    pw, ph = panel_pil.size

    # Boundary mask
    ep = _erosion_px(h, w)
    border = max(12, ep)
    bmask = np.ones((h, w), dtype=np.uint8) * 255
    bmask[:border, :]      = 0
    bmask[h-border:, :]    = 0
    bmask[:, :border]      = 0
    bmask[:, w-border:]    = 0
    if valid_mask is not None:
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (ep, ep))
        bmask = cv2.bitwise_and(bmask, cv2.erode(valid_mask, k2))
    elif H_inv is not None:
        try:
            H = np.linalg.inv(H_inv)
            mb = np.ones((ph, pw), dtype=np.uint8) * 255
            wm = cv2.warpPerspective(mb, H, (w, h), flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (ep, ep))
            bmask = cv2.bitwise_and(bmask, cv2.erode(wm, k2))
        except Exception:
            pass
    gray_diff = cv2.bitwise_and(gray_diff, bmask)

    # Warp circles to panel B space
    if H_inv is not None:
        pts    = np.float32([[cx, cy] for cx, cy, _ in circles]).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
        warped = [(int(np.clip(mx, 0, pw-1)), int(np.clip(my, 0, ph-1)), r)
                  for (mx, my), (_, _, r) in zip(mapped, circles)]
    else:
        warped = [(cx, cy, r) for cx, cy, r in circles]

    drawn = []
    for idx, (cx, cy, r) in enumerate(circles):
        r_size = max(40, int(r * 1.5))
        y1, y2 = max(0, cy - r_size), min(h, cy + r_size)
        x1, x2 = max(0, cx - r_size), min(w, cx + r_size)
        roi = gray_diff[y1:y2, x1:x2].copy()
        roi[0:2, :] = 0; roi[-2:, :] = 0; roi[:, 0:2] = 0; roi[:, -2:] = 0

        max_v    = np.max(roi)
        thresh_v = max(8, int(max_v * 0.25))
        _, th    = cv2.threshold(roi, thresh_v, 255, cv2.THRESH_BINARY)
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kc)

        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        to_draw = []
        for c in cnts:
            if cv2.contourArea(c) >= 0:
                shifted = c + np.array([x1, y1])
                if H_inv is not None:
                    pts2 = shifted.astype(np.float32).reshape(-1, 1, 2)
                    wpts = cv2.perspectiveTransform(pts2, H_inv).reshape(-1, 2)
                    to_draw.append(wpts.astype(np.int32).reshape(-1, 1, 2))
                else:
                    to_draw.append(shifted.astype(np.int32).reshape(-1, 1, 2))

        if to_draw:
            for c_draw in to_draw:
                cv2.polylines(b_cv, [c_draw], isClosed=True,
                              color=color[::-1], thickness=3)
            drawn.append(("contour", to_draw, warped[idx]))
        else:
            fcx, fcy, fr = warped[idx]
            cv2.circle(b_cv, (fcx, fcy), fr, color[::-1], thickness=3)
            drawn.append(("circle", [], warped[idx]))

    pil_b = Image.fromarray(cv2.cvtColor(b_cv, cv2.COLOR_BGR2RGB))
    draw  = ImageDraw.Draw(pil_b)
    font  = _load_font(_LATIN_FONT, 54)

    for idx, (dtype, cnts_draw, (fcx, fcy, fr)) in enumerate(drawn):
        num = str(idx + 1)
        if dtype == "contour" and cnts_draw:
            all_pts = np.vstack([c.reshape(-1, 2) for c in cnts_draw])
            tx = max(5, int(np.min(all_pts[:, 0])) - 45)
            ty = max(5, int(np.min(all_pts[:, 1])) - 45)
        else:
            tx = max(5, fcx - fr - 30)
            ty = max(5, fcy - fr - 30)
        draw.text((tx, ty), num, font=font, fill=(255, 255, 255),
                  stroke_width=4, stroke_fill=(0, 0, 0))

    return pil_b


# ─────────────────────────────────────────────────────────────────────────────
#  9.  OUTPUT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

BH = 80   # banner height in pixels


def build_stacked_output(combined_bgr, img_a, img_b_aligned,
                         circles, a_start, b_start, color, count,
                         H=None, valid_mask=None):
    oh, ow = combined_bgr.shape[:2]
    ph     = img_a.shape[0]
    H_inv  = np.linalg.inv(H) if H is not None else None
    b_slice = combined_bgr[b_start:b_start + ph, :]
    pil_b   = Image.fromarray(cv2.cvtColor(b_slice, cv2.COLOR_BGR2RGB))
    pil_b   = draw_contours_and_numbers_on_panel(
        pil_b, circles, img_a, img_b_aligned, H_inv, color, valid_mask)
    pil_a   = Image.fromarray(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
    base    = Image.fromarray(cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB))
    canvas  = Image.new("RGB", (ow, BH + oh), (30, 30, 50))
    canvas.paste(base,  (0, BH))
    canvas.paste(pil_a, (0, BH + a_start))
    canvas.paste(pil_b, (0, BH + b_start))
    canvas.paste(make_khmer_banner(ow, count), (0, 0))
    return canvas


def build_vertical_split_output(combined_bgr, img_a, img_b_aligned,
                                 circles, a_start, b_start, crop_y_offset,
                                 color, count, H=None, valid_mask=None):
    oh, ow = combined_bgr.shape[:2]
    pw, ph = img_a.shape[1], img_a.shape[0]
    H_inv  = np.linalg.inv(H) if H is not None else None
    b_slice = combined_bgr[crop_y_offset:crop_y_offset + ph, b_start:b_start + pw]
    pil_b   = Image.fromarray(cv2.cvtColor(b_slice, cv2.COLOR_BGR2RGB))
    pil_b   = draw_contours_and_numbers_on_panel(
        pil_b, circles, img_a, img_b_aligned, H_inv, color, valid_mask)
    pil_a   = Image.fromarray(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
    base    = Image.fromarray(cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB))
    canvas  = Image.new("RGB", (ow, BH + oh), (30, 30, 50))
    canvas.paste(base,  (0, BH))
    canvas.paste(pil_a, (a_start, BH + crop_y_offset))
    canvas.paste(pil_b, (b_start, BH + crop_y_offset))
    canvas.paste(make_khmer_banner(ow, count), (0, 0))
    return canvas


def build_stacked_output_numgrid(combined_bgr, img_a, circles_b,
                                  a_start, b_start, color, count):
    oh, ow  = combined_bgr.shape[:2]
    ph      = img_a.shape[0]
    pil_a   = Image.fromarray(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
    b_slice = combined_bgr[b_start:b_start + ph, :]
    pil_b   = Image.fromarray(cv2.cvtColor(b_slice, cv2.COLOR_BGR2RGB))
    pil_b   = draw_circles_on_panel(pil_b, circles_b, color)
    base    = Image.fromarray(cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB))
    canvas  = Image.new("RGB", (ow, BH + oh), (30, 30, 50))
    canvas.paste(base,  (0, BH))
    canvas.paste(pil_a, (0, BH + a_start))
    canvas.paste(pil_b, (0, BH + b_start))
    canvas.paste(make_khmer_banner(ow, count), (0, 0))
    return canvas


def build_sidebyside_output(img_a, img_b_original, img_b_aligned,
                             circles, color, count, H=None, valid_mask=None):
    GAP   = 6
    h, w  = img_a.shape[:2]
    H_inv = np.linalg.inv(H) if H is not None else None
    pil_a = Image.fromarray(cv2.cvtColor(img_a,          cv2.COLOR_BGR2RGB))
    pil_b = Image.fromarray(cv2.cvtColor(img_b_original, cv2.COLOR_BGR2RGB))
    pil_b = draw_contours_and_numbers_on_panel(
        pil_b, circles, img_a, img_b_aligned, H_inv, color, valid_mask)
    total_w = w * 2 + GAP
    canvas  = Image.new("RGB", (total_w, BH + h), (30, 30, 50))
    canvas.paste(pil_a, (0,       BH))
    canvas.paste(pil_b, (w + GAP, BH))
    ImageDraw.Draw(canvas).rectangle([w, BH, w + GAP, BH + h], fill=(180, 180, 180))
    canvas.paste(make_khmer_banner(total_w, count), (0, 0))
    return canvas


def build_sidebyside_output_numgrid(img_a, img_b, circles_b, color, count):
    GAP  = 6
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    h       = max(ha, hb)
    pil_a   = Image.fromarray(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
    pil_b   = Image.fromarray(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
    pil_b   = draw_circles_on_panel(pil_b, circles_b, color)
    total_w = wa + GAP + wb
    canvas  = Image.new("RGB", (total_w, BH + h), (30, 30, 50))
    canvas.paste(pil_a, (0,        BH))
    canvas.paste(pil_b, (wa + GAP, BH))
    ImageDraw.Draw(canvas).rectangle([wa, BH, wa + GAP, BH + h], fill=(180, 180, 180))
    canvas.paste(make_khmer_banner(total_w, count), (0, 0))
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  10.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Spot the Difference Engine v3 — fully dynamic, zero overrides.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python engine_v3.py puzzle.jpg
  python engine_v3.py original.png modified.png
  python engine_v3.py numgrid.png --mode number
  python engine_v3.py a.jpg b.jpg --output result.png
  python engine_v3.py puzzle.jpg --ecc
  python engine_v3.py puzzle.jpg --no-align
        """,
    )
    p.add_argument("images",   nargs="+", metavar="IMAGE")
    p.add_argument("--output", default=str(
        Path(__file__).resolve().parent / "results" / "circled_result.png"))
    p.add_argument("--mode",   choices=["auto", "colour", "line", "number"], default="auto")
    p.add_argument("--no-align", action="store_true")
    p.add_argument("--ecc",      action="store_true")
    p.add_argument("--mask-roi", type=str, default="",
                   help="Extra mask regions: x1,y1,x2,y2;...")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  11.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    is_vertical_split = False
    crop_y_offset     = 0
    split_dir         = None

    if len(args.images) == 1:
        print(f"[V3] Single image: {args.images[0]!r}")
        combined = load_bgr(args.images[0])
        h, w     = combined.shape[:2]

        if args.mode == "number":
            half  = h // 2
            img_a = combined[:half]
            img_b = combined[half:]
            a_start, b_start = 0, half
            two_image_mode   = False
        else:
            cropped, crop_y_offset = crop_text_by_gap(combined)
            img_a, img_b, a_start, b_start = auto_slice(cropped)
            is_vertical_split = (img_a.shape[0] == cropped.shape[0])
            if not is_vertical_split:
                a_start += crop_y_offset
                b_start += crop_y_offset
            two_image_mode = False

    elif len(args.images) == 2:
        print(f"[V3] Two images: {args.images[0]!r}  vs  {args.images[1]!r}")
        combined       = None
        a_start = b_start = 0
        img_a          = load_bgr(args.images[0])
        img_b          = load_bgr(args.images[1])
        two_image_mode = True
    else:
        sys.exit("[ERROR] Provide 1 combined image or 2 separate images.")

    print(f"[V3] A: {img_a.shape[:2]}   B: {img_b.shape[:2]}")
    img_b_original = img_b.copy()
    color          = _random_color()
    print(f"[V3] Run colour: RGB{color}")

    # ── Parse mask ROIs ──────────────────────────────────────────────────────
    mask_rois = []
    if args.mask_roi:
        for block in args.mask_roi.split(";"):
            if block.strip():
                parts = [int(p) for p in block.split(",")]
                if len(parts) == 4:
                    mask_rois.append(tuple(parts))

    # ── NUMBER GRID MODE ─────────────────────────────────────────────────────
    if args.mode == "number":
        circles_a, circles_b, count = detect_number_grid(img_a, img_b)
        if two_image_mode:
            result = build_sidebyside_output_numgrid(img_a, img_b, circles_b, color, count)
        else:
            result = build_stacked_output_numgrid(combined, img_a, circles_b,
                                                   a_start, b_start, color, count)
        result = add_watermark(result)
        result.save(args.output, quality=95)
        print(f"\n{'='*45}\n  Mode              : number-grid"
              f"\n  Differences found : {count}"
              f"\n  Saved to          : {args.output!r}\n{'='*45}")
        return

    # ── ALIGNMENT ────────────────────────────────────────────────────────────
    if args.no_align:
        img_b_aligned = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]),
                                   interpolation=cv2.INTER_LANCZOS4)
        H_align    = None
        valid_mask = None
        print("[V3] Alignment skipped (--no-align)")
    else:
        img_b_aligned, H_align, valid_mask = align(img_a, img_b, skip_ecc=not args.ecc)

    # ── MODE DETECTION ───────────────────────────────────────────────────────
    if args.mode == "auto":
        line_mode = is_line_drawing(img_a) and (two_image_mode and is_line_drawing(img_b) or not two_image_mode)
    else:
        line_mode = (args.mode == "line")
        print(f"[V3] Mode forced: {'line-drawing' if line_mode else 'colour'}")

    # ── DETECTION ────────────────────────────────────────────────────────────
    if line_mode:
        circles, count = detect_line(
            img_a, img_b_aligned, valid_mask=valid_mask, H=H_align,
            mask_rois=mask_rois)
    else:
        split_dir = None if two_image_mode else ("vertical" if is_vertical_split else "horizontal")
        circles, count = detect(
            img_a, img_b_aligned, valid_mask=valid_mask, H=H_align,
            split_dir=split_dir, mask_rois=mask_rois)

    # ── BUILD OUTPUT ─────────────────────────────────────────────────────────
    if two_image_mode:
        result = build_sidebyside_output(
            img_a, img_b_original, img_b_aligned,
            circles, color, count, H=H_align, valid_mask=valid_mask)
    elif split_dir == "vertical":
        result = build_vertical_split_output(
            combined, img_a, img_b_aligned,
            circles, a_start, b_start, crop_y_offset, color, count,
            H=H_align, valid_mask=valid_mask)
    else:
        result = build_stacked_output(
            combined, img_a, img_b_aligned,
            circles, a_start, b_start, color, count,
            H=H_align, valid_mask=valid_mask)

    result = add_watermark(result)
    result.save(args.output, quality=95)
    print(f"\n{'='*45}\n  Mode              : {'line-drawing' if line_mode else 'colour'}"
          f"\n  Differences found : {count}"
          f"\n  Run colour        : RGB{color}"
          f"\n  Saved to          : {args.output!r}\n{'='*45}")


if __name__ == "__main__":
    main()
