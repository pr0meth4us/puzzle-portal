import sys
import numpy as np
import cv2

sys.path.append("/Users/nicksng/code/puzzle-portal/spot_the_difference")
import spot_the_differences as std
import scratch.eval_puz7_8 as ev

c_8, ec_8, col_8, max_r_8 = ev.precompute_candidates("/Users/nicksng/code/puzzle-portal/spot_the_difference/puzzles/puzzle_08.jpg", "Puzzle 8")

delta_floors = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 25.0, 30.0]
min_areas = [20, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300]
merge_radii = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80]

matching_configs = []

for ma in min_areas:
    filtered_cnts = [c for c in c_8 if c.area >= ma]
    all_candidates = filtered_cnts + ec_8
    if not all_candidates:
        continue
    deltas = sorted(c.delta for c in all_candidates)
    
    for df in delta_floors:
        floor = 10.0 if (df == 7.0 and col_8) else df
        threshold, _ = std._auto_threshold(deltas, floor)
        surviving = [c for c in all_candidates if c.delta >= threshold]
        
        for mr in merge_radii:
            groups = []
            for c in surviving:
                cx, cy, r, delta = float(c.cx), float(c.cy), c.r, c.delta
                merged = False
                for grp in groups:
                    if ((cx - grp[2]) ** 2 + (cy - grp[3]) ** 2) ** 0.5 < mr:
                        grp[0].append((cx, cy, r))
                        grp[1] = max(grp[1], delta)
                        grp[2] = float(np.mean([s[0] for s in grp[0]]))
                        grp[3] = float(np.mean([s[1] for s in grp[0]]))
                        merged = True
                        break
                if not merged:
                    groups.append([[(cx, cy, r)], delta, cx, cy])
            
            if len(groups) >= 3:
                all_d = np.array([g[1] for g in groups])
                keep = []
                for grp in groups:
                    others = all_d[all_d != grp[1]]
                    med_oth = float(np.median(others)) if len(others) else grp[1]
                    if grp[1] >= std.LOW_DELTA_FRAC * med_oth:
                        keep.append(grp)
                groups = keep
            
            if len(groups) == 10:
                matching_configs.append((df, ma, mr))

print(f"Total matching configs for Puzzle 8: {len(matching_configs)}")
for cfg in matching_configs:
    print(f"  delta_floor={cfg[0]}, min_area={cfg[1]}, merge_radius={cfg[2]}")
