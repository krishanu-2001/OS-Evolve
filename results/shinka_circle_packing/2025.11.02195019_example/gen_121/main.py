# EVOLVE-BLOCK-START
"""Hybrid ring+hex constructor with force relaxation and annealing for n=26 circles"""

import numpy as np

def construct_packing():
    """
    Hybrid initialization: structured rings + hex grid fill,
    followed by force-based relaxation and adaptive simulated annealing.
    Returns:
        centers: np.array (26,2)
        radii:   np.array (26,)
    """
    np.random.seed(42)
    n = 26

    centers = np.zeros((n,2))
    # 1. Place a central circle
    centers[0] = [0.5, 0.5]
    # 2. Place 8 in a main ring (radius chosen for good internal clearance, randomized)
    main_ring_r = 0.29 + np.random.uniform(-0.02, 0.02)
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i+1] = [0.5 + main_ring_r * np.cos(angle), 0.5 + main_ring_r * np.sin(angle)]

    # 3. Place 4 in corners, ensure inside
    corners = np.array([[0.05,0.05], [0.05,0.95], [0.95,0.05], [0.95,0.95]])
    centers[9:13] = corners

    # 4. The next 8: edge-centered (well-inside border)
    edge_offset = 0.09
    edges = np.array([
        [edge_offset,0.5], [0.5,edge_offset], [1-edge_offset,0.5], [0.5,1-edge_offset],
        [edge_offset,edge_offset],
        [1-edge_offset,edge_offset],
        [edge_offset,1-edge_offset],
        [1-edge_offset,1-edge_offset]
    ])
    centers[13:21] = edges

    # 5. Fill remaining positions (5 left) using a tightest-packed hex grid determination
    used = 21
    needed = n - used
    hex_candidates = []
    grid_m = 8
    dx = 0.90/(grid_m-1)
    dy = dx * np.sqrt(3)/2
    for i in range(grid_m):
        for j in range(grid_m):
            x = 0.05 + j*dx + (i%2)*(dx/2)
            y = 0.05 + i*dy
            if 0.04 < x < 0.96 and 0.04 < y < 0.96:
                hex_candidates.append([x,y])
    hex_candidates = np.array(hex_candidates)
    # Exclude close points to placed circles
    dists = np.linalg.norm(hex_candidates[:,None,:] - centers[:used][None,:,:], axis=2).min(axis=1)
    idx = np.argsort(-dists)[:needed]
    centers[used:used+needed] = hex_candidates[idx]

    # 6. Add slight random jitter to all points (from physics code idea)
    centers += (np.random.rand(n,2) - 0.5)*0.018
    centers = np.clip(centers, 0.015, 0.985)

    # Step 2: Initial radii assignment
    radii = compute_max_radii(centers)

    # Step 3: Physics-inspired force-based overlap/border relaxation
    alpha = 0.035
    for it in range(400):
        forces = np.zeros((n,2))
        # Repulsion (overlap)
        for i in range(n):
            for j in range(i+1,n):
                dxy = centers[i] - centers[j]
                dist = np.hypot(dxy[0], dxy[1]) + 1e-9
                allow = radii[i]+radii[j]
                if dist < allow and dist > 1e-10:
                    # push away nearly proportionally to overlap
                    overlap = (allow - dist) / (dist+1e-12)
                    f = dxy * overlap * 0.5
                    forces[i] += f
                    forces[j] -= f
        # Border corrective
        margin = 0.001
        for i in range(n):
            x,y = centers[i]
            r = radii[i]
            # left/right
            if x - r < 0:
                forces[i,0] += (r - x + margin)
            if x + r > 1:
                forces[i,0] -= (x + r - 1 + margin)
            if y - r < 0:
                forces[i,1] += (r - y + margin)
            if y + r > 1:
                forces[i,1] -= (y + r - 1 + margin)
        # Update
        centers += alpha * forces
        centers = np.clip(centers, 0.01, 0.99)
        # Relax radii & decay step
        radii = compute_max_radii(centers)
        alpha *= 0.992 if it < 250 else 0.998

    # Step 4: Simulated annealing with adaptive temperature and forced multi-point wiggles
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    T0 = 0.0025
    temp = T0
    stagnation = 0
    for k in range(1500):
        # select random 1 or 2 circles to jitter (diversity mutator)
        kchg = 1 if np.random.rand() < 0.75 else 2
        idxs = np.random.choice(n,kchg,replace=False)
        old_pos = centers[idxs].copy()
        delta = (np.random.randn(kchg,2)) * (0.008 * temp/(T0+1e-9))
        centers[idxs] = np.clip(centers[idxs]+delta, 0.01, 0.99)
        radii_tmp = compute_max_radii(centers)
        sum_new = np.sum(radii_tmp)
        dE = sum_new - best_sum
        if dE > 0 or np.random.rand() < np.exp(dE/(temp+1e-12)):
            if dE > 1e-8:
                best_sum = sum_new
                best_radii = radii_tmp.copy()
                best_centers = centers.copy()
                stagnation = 0
            else:
                stagnation += 1
        else:
            centers[idxs] = old_pos
            stagnation += 1
        # Adaptive cooling/heating
        if stagnation > 60:
            temp = min(T0, temp*1.2)
            stagnation = 0
        else:
            temp = max(T0*0.15, temp*0.997)
    centers, radii = best_centers, best_radii
    # Stage 5: local greedy repacking sweep to further improve individual radii
    def _greedy_repack_stage(c, samples=150):
        n = c.shape[0]
        for i in range(n):
            # current radii and other circles
            radii_all = compute_max_radii(c)
            others = np.delete(c, i, axis=0)
            other_rs = np.delete(radii_all, i)
            best_p = c[i].copy()
            best_r = radii_all[i]
            # Adaptive sampling: more samples for smaller circles
            # (smaller radii = more crowded, need more search)
            min_r, max_r = radii_all.min(), radii_all.max()
            # Map radius to [0,1], invert so small radii get more samples
            if max_r > min_r:
                rel = (best_r - min_r) / (max_r - min_r)
            else:
                rel = 0.5
            # At least 200, up to 800 samples
            n_local = int(200 + 600 * (1 - rel))
            n_global = int(100 + 300 * (1 - rel))
            # generate candidate points: local jitter and some global samples
            local_pts = best_p + np.random.randn(n_local, 2) * (best_r * 0.4 + 1e-6)
            global_pts = np.random.rand(n_global, 2) * 0.98 + 0.01
            pts = np.vstack((local_pts, global_pts))
            pts = np.clip(pts, 0.01, 0.99)
            for p in pts:
                # wall constraint
                r_new = min(p[0], 1-p[0], p[1], 1-p[1])
                # overlap constraint
                if others.size > 0:
                    d = np.linalg.norm(others - p, axis=1) - other_rs
                    r_new = min(r_new, d.min())
                if r_new > best_r:
                    best_r = r_new
                    best_p = p
            c[i] = best_p
        return c
    centers = _greedy_repack_stage(centers)
    radii = compute_max_radii(centers)
    return centers, radii

def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.
    Uses a robust pairwise relaxation (many more iterations than parent codes).
    """
    n = centers.shape[0]
    xs, ys = centers[:,0], centers[:,1]
    radii = np.minimum.reduce([xs, ys, 1-xs, 1-ys])
    for iter in range(70):
        max_change = 0.0
        for i in range(n):
            for j in range(i+1,n):
                dxy = centers[i] - centers[j]
                dist = np.hypot(dxy[0], dxy[1])
                max_sum = radii[i] + radii[j]
                if max_sum > dist and dist > 1e-10:
                    scale = dist / (max_sum+1e-12)
                    old_i, old_j = radii[i], radii[j]
                    radii[i] *= scale
                    radii[j] *= scale
                    max_change = max(max_change, abs(radii[i]-old_i), abs(radii[j]-old_j))
        if max_change < 1e-7:
            break
    return radii
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii