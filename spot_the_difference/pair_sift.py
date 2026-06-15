import os
import cv2
from pathlib import Path

puzzles_dir = Path('/Users/nicksng/code/random/spot_the_difference/puzzles')
answers_dir = Path('/Users/nicksng/code/random/spot_the_difference/correct answers')

puzzles = [p for p in puzzles_dir.glob('*.*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
answers = [a for a in answers_dir.glob('*.*') if a.suffix.lower() in ['.jpg', '.jpeg', '.png']]

det = cv2.SIFT_create(nfeatures=5000)

print("Extracting features...")
p_feats = {}
for p in puzzles:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    img = cv2.resize(img, (600, 600))
    kp, des = det.detectAndCompute(img, None)
    p_feats[p] = des

a_feats = {}
for a in answers:
    img = cv2.imread(str(a), cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    img = cv2.resize(img, (600, 600))
    kp, des = det.detectAndCompute(img, None)
    a_feats[a] = des

print("Matching...")
bf = cv2.BFMatcher()
pairs = []
for a_path, a_des in a_feats.items():
    best_match = None
    best_score = 0
    for p_path, p_des in p_feats.items():
        if a_des is None or p_des is None: continue
        try:
            matches = bf.knnMatch(a_des, p_des, k=2)
        except Exception:
            continue
        good = 0
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good += 1
        if good > best_score:
            best_score = good
            best_match = p_path
    pairs.append((a_path, best_match, best_score))

print("\nPAIRS:")
for a, p, s in pairs:
    print(f"ANSWER: {a.name} -> PUZZLE: {p.name if p else 'NONE'} (score: {s})")
