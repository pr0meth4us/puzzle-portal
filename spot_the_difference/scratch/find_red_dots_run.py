import cv2
import numpy as np

def find_red_dots_by_alignment(puz_path, ans_path, name):
    puz = cv2.imread(puz_path)
    ans = cv2.imread(ans_path)
    if puz is None or ans is None:
        print(f"Error loading {puz_path} or {ans_path}")
        return
        
    print(f"\n==================== Aligning {name} ====================")
    print(f"Puzzle shape: {puz.shape}, Answer shape: {ans.shape}")
    
    # Use SIFT to align puzzle to answer key
    sift = cv2.SIFT_create(nfeatures=5000)
    kp_puz, des_puz = sift.detectAndCompute(puz, None)
    kp_ans, des_ans = sift.detectAndCompute(ans, None)
    
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = matcher.knnMatch(des_puz, des_ans, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    if len(good) < 10:
        print("Not enough SIFT matches to align!")
        return
        
    src_pts = np.float32([kp_puz[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ans[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Warp puzzle to match answer key
    h_ans, w_ans = ans.shape[:2]
    puz_warped = cv2.warpPerspective(puz, H, (w_ans, h_ans))
    
    # Now compute the difference between answer key and warped puzzle
    diff = cv2.absdiff(ans, puz_warped)
    
    # Red signal: R channel is high in diff, and much higher than G and B
    diff_bgr = diff.astype(np.float32)
    b_diff, g_diff, r_diff = diff_bgr[:, :, 0], diff_bgr[:, :, 1], diff_bgr[:, :, 2]
    red_signal = r_diff - np.maximum(g_diff, b_diff)
    
    _, thresh = cv2.threshold(red_signal, 70, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    
    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dots_in_ans = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 10:
            (cx, cy), r = cv2.minEnclosingCircle(c)
            dots_in_ans.append((cx, cy, r, area))
            
    # Filter close dots in answer key
    filtered_ans = []
    for cx, cy, r, area in dots_in_ans:
        too_close = False
        for kx, ky, _, _ in filtered_ans:
            if np.hypot(cx - kx, cy - ky) < 25:
                too_close = True
                break
        if not too_close:
            filtered_ans.append((cx, cy, r, area))
            
    # Now map the filtered dots back to the puzzle's coordinate space
    H_inv = np.linalg.inv(H)
    dots_in_puz = []
    for cx, cy, r, area in filtered_ans:
        pt = np.array([cx, cy, 1.0]).reshape(3, 1)
        mapped_pt = H_inv @ pt
        mx = mapped_pt[0, 0] / mapped_pt[2, 0]
        my = mapped_pt[1, 0] / mapped_pt[2, 0]
        dots_in_puz.append((mx, my, r, cx, cy))
        
    print(f"Found {len(dots_in_puz)} red dots:")
    # Sort by my
    dots_in_puz.sort(key=lambda x: x[1])
    for i, (mx, my, r, cx, cy) in enumerate(dots_in_puz):
        print(f"  Dot {i+1}: In Puzzle = ({mx:.1f}, {my:.1f}), In Answer = ({cx:.1f}, {cy:.1f}), r={r:.1f}")

find_red_dots_by_alignment("puzzles/puzzle_07.jpg", "correct_answers/answer_07.jpg", "Puzzle 7")
find_red_dots_by_alignment("puzzles/puzzle_08.jpg", "correct_answers/answer_08.jpg", "Puzzle 8")
