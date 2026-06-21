import numpy as np

scores = [
    53.55, 45.64, 43.74, 43.74, 40.41, 39.24, 38.54, 37.73, 37.60, 37.32,
    36.76, 34.23, 32.14, 29.47, 28.76, 28.57, 28.36, 27.29, 26.63, 21.92,
    21.50, 20.73, 18.33, 18.29, 16.66, 15.97, 15.91, 14.94, 14.27, 14.25,
    13.00, 12.00, 11.50, 11.00, 10.50, 10.00, 9.50, 9.00, 8.50, 8.00
]
# Add some more low scores to simulate 90 cells
scores += [8.0] * 50

s = np.array(scores)
c1 = float(np.percentile(s, 25))
c2 = float(np.percentile(s, 75))

for _ in range(20):
    g1 = s[np.abs(s - c1) < np.abs(s - c2)]
    g2 = s[np.abs(s - c1) >= np.abs(s - c2)]
    if len(g1) == 0 or len(g2) == 0:
        break
    new_c1 = float(np.mean(g1))
    new_c2 = float(np.mean(g2))
    if new_c1 == c1 and new_c2 == c2:
        break
    c1, c2 = new_c1, new_c2

t = (np.max(g1) + np.min(g2)) / 2.0
print(f"Centroids: c1={c1:.2f}, c2={c2:.2f}")
print(f"Threshold: {t:.2f}")
print(f"Count of differences (>= threshold): {len(g2)}")
