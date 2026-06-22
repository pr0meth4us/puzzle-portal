import cv2
import numpy as np
import sys

sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std

# Define correct ground truth coordinates
circles = [
    (453, 210, 28),   # 1
    (617, 503, 28),   # 2
    (211, 565, 28),   # 3
    (408, 792, 28),   # 4
    (953, 812, 28),   # 5
    (387, 857, 28),   # 6
    (633, 871, 28),   # 7
    (296, 900, 28),   # 8
    (809, 933, 28),   # 9
    (46, 458, 28),    # 10
    (45, 539, 28),    # 11
    (1095, 449, 28)   # 12
]

combined = std.load_bgr("puzzles/puzzle_07.jpg")
cropped, y_off = std.crop_text_by_gap(combined)
img_a, img_b, a_start, b_start = std.auto_slice(cropped)
img_b_aligned, valid_y, H = std.align(img_a, img_b, skip_ecc=False)

h, w = img_a.shape[:2]
H_inv = np.linalg.inv(H) if H is not None else None
diff = cv2.absdiff(img_a, img_b_aligned)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

bmask = np.ones((h, w), dtype=np.uint8) * 255
bmask[:, :40] = 0
bmask[:, w - 40:] = 0
bmask[:32, :] = 0
bmask[h - 32:, :] = 0
bgr_sum = np.sum(img_b_aligned, axis=2)
bmask[bgr_sum < 15] = 0
gray_diff = cv2.bitwise_and(gray_diff, bmask)

for idx, (cx, cy, r) in enumerate(circles):
    r_size = max(40, int(r * 1.5))
    y1, y2 = max(0, cy - r_size), min(h, cy + r_size)
    x1, x2 = max(0, cx - r_size), min(w, cx + r_size)
    roi = gray_diff[y1:y2, x1:x2].copy()
    roi[0:2, :] = 0; roi[-2:, :] = 0; roi[:, 0:2] = 0; roi[:, -2:] = 0
    max_val = np.max(roi)
    thresh_val = max(8, int(max_val * 0.25))
    _, th = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k_close)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_to_draw = []
    if cnts:
        for c in cnts:
            if cv2.contourArea(c) >= 4:
                contour_shifted = c + np.array([x1, y1])
                if H_inv is not None:
                    pts_pts = contour_shifted.astype(np.float32).reshape(-1, 1, 2)
                    warped_pts = cv2.perspectiveTransform(pts_pts, H_inv).reshape(-1, 2)
                    contour_draw = warped_pts.astype(np.int32)
                else:
                    contour_draw = contour_shifted.astype(np.int32)
                contours_to_draw.append(contour_draw.reshape(-1, 1, 2))
                
    if contours_to_draw:
        all_pts = np.vstack([c.reshape(-1, 2) for c in contours_to_draw])
        x_min = int(np.min(all_pts[:, 0]))
        y_min = int(np.min(all_pts[:, 1]))
        tx = max(5, x_min - 45)
        ty = max(5, y_min - 45)
        print(f"Diff {idx+1}: Contours count={len(contours_to_draw)}, x_min={x_min}, y_min={y_min}, tx={tx}, ty={ty}")
    else:
        pts = np.float32([[cx, cy]]).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
        mcx, mcy = int(mapped[0][0]), int(mapped[0][1])
        tx = max(5, mcx - r - 30)
        ty = max(5, mcy - r - 30)
        print(f"Diff {idx+1}: Circle fallback, mcx={mcx}, mcy={mcy}, tx={tx}, ty={ty}")
