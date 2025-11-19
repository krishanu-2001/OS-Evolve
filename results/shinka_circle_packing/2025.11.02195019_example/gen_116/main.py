# EVOLVE-BLOCK-START
"""Hybrid adaptive simulated annealing and force-based packing for 26 circles"""

import numpy as np

def construct_packing():
    """
    Construct and optimize 26 circles packing in unit square via 
    strategic initialization + force relaxation + adaptive SA + local refinement.
    
    Returns:
        centers: (26,2) array of circle centers
        radii: (26,) array of circle radii
    """
    np.random.seed(42)
    n = 26

    # --- 1) Strategic initialization ---
    centers = np.zeros((n,2))

    # Large corner circles (4)
    corner_r = 0.12
    corners = np.array([
        [corner_r, corner_r],
        [1 - corner_r, corner_r],
        [corner_r, 1 - corner_r],
        [1 - corner_r, 1 - corner_r]
    ])
    centers[0:4] = corners

    # Medium edge circles (8)
    edge_r = 0.07
    edge_positions = []
    xs = np.linspace(corner_r*2, 1 - corner_r*2, 4)
    # bottom edge
    edge_positions.extend([[x, edge_r] for x in xs])
    # top edge
    edge_positions.extend([[x, 1 - edge_r] for x in xs])
    # left and right edges (4 + 4)
    ys = np.linspace(corner_r*2, 1 - corner_r*2, 4)
    left_right_edges = [[edge_r, y] for y in ys] + [[1 - edge_r, y] for y in ys]
    # Assign medium edges to centers indices 4:12 + next 8
    centers[4:12] = np.array(edge_positions)
    centers = np.vstack([centers[:12], np.array(left_right_edges)])

    # Remaining centers filled with hex lattice in center square [0.2,0.8]^2
    remaining = n - centers.shape[0]
    m = int(np.ceil(np.sqrt(remaining / 0.866)))  # hex grid rows estimate
    dx = 0.6 / (m - 1)
    dy = dx * np.sqrt(3) / 2
    hex_pts = []
    for i in range(m):
        for j in range(m):
            x = 0.2 + j * dx + (i % 2) * (dx / 2)
            y = 0.2 + i * dy
            if x <= 0.8 and y <= 0.8:
                hex_pts.append((x, y))
    hex_pts = np.array(hex_pts)
    center_point = np.array([0.5,0.5])
    d_center = np.linalg.norm(hex_pts - center_point, axis=1)
    idx = np.argsort(d_center)[:remaining]
    centers = np.vstack([centers, hex_pts[idx]])

    # Add small jitter to break symmetry
    centers += (np.random.rand(n, 2) - 0.5) * 0.01
    centers = np.clip(centers, 0.01, 0.99)

    # --- 2) Force-based relaxation ---
    radii = compute_max_radii(centers)
    alpha = 0.02
    for it in range(600):
        forces = np.zeros((n,2))

        # Pairwise repulsion for overlaps
        for i in range(n):
            for j in range(i+1, n):
                dxy = centers[i] - centers[j]
                dist = np.hypot(*dxy) + 1e-10
                allow = radii[i] + radii[j]
                if dist < allow:
                    overlap = (allow - dist) / dist
                    forces[i] += dxy * overlap
                    forces[j] -= dxy * overlap

        # Border corrective forces
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0: forces[i,0] += (r - x)
            if x + r > 1: forces[i,0] -= (x + r - 1)
            if y - r < 0: forces[i,1] += (r - y)
            if y + r > 1: forces[i,1] -= (y + r - 1)

        centers += alpha * forces
        centers = np.clip(centers, 0.01, 0.99)
        radii = compute_max_radii(centers)
        alpha *= 0.995

    # --- 3) Adaptive simulated annealing ---
    centers_sa = centers.copy()
    radii_sa = radii.copy()
    sum_sa = radii_sa.sum()
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = sum_sa

    T = 0.001
    cooling_rate = 0.995
    max_T = 0.01
    no_improve = 0
    rng = np.random.default_rng(1234)

    for k in range(1500):
        # Multi-circle moves every 100 iters, otherwise single move
        if k % 100 == 0:
            count = rng.integers(2,5)
            idxs = rng.choice(n, size=count, replace=False)
        else:
            idxs = [rng.integers(n)]
        old_pos = centers_sa[idxs].copy()
        scale = 0.006 * (1 - k/1500)
        perturb = rng.normal(0, scale, size=(len(idxs),2))
        centers_sa[idxs] = np.clip(centers_sa[idxs]+perturb, 0.01, 0.99)
        radii_tmp = compute_max_radii(centers_sa)
        sum_new = np.sum(radii_tmp)
        dE = sum_new - sum_sa
        accept = False
        if dE > 0:
            accept = True
        else:
            p_accept = np.exp(dE / (T + 1e-15))
            if rng.random() < p_accept:
                accept = True
        if accept:
            sum_sa = sum_new
            radii_sa = radii_tmp
            no_improve = 0
            if sum_new > best_sum + 1e-8:
                best_sum = sum_new
                best_centers = centers_sa.copy()
                best_radii = radii_tmp.copy()
        else:
            centers_sa[idxs] = old_pos
            no_improve += 1
        # Adaptive temperature control
        if no_improve > 100:
            T = min(T*1.1, max_T)
            no_improve = 0
        else:
            T *= cooling_rate

    centers = best_centers
    radii = best_radii

    # --- 4) Local greedy refinement ---
    directions = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]])
    for i in range(n):
        orig_pos = centers[i].copy()
        for step in [0.01, 0.005, 0.0025, 0.001]:
            improved = True
            while improved:
                improved = False
                for d in directions:
                    candidate = orig_pos + d * step
                    candidate = np.clip(candidate, 0.01, 0.99)
                    centers[i] = candidate
                    candidate_radii = compute_max_radii(centers)
                    if np.sum(candidate_radii) > np.sum(radii) + 1e-12:
                        radii = candidate_radii
                        orig_pos = candidate
                        improved = True
                centers[i] = orig_pos

        # Localized dense perturbations around current best pos
        local_scale = step / 5
        for _ in range(20):
            delta = (np.random.rand(2) - 0.5) * 2 * local_scale
            candidate = np.clip(orig_pos + delta, 0.01, 0.99)
            centers[i] = candidate
            candidate_radii = compute_max_radii(centers)
            if np.sum(candidate_radii) > np.sum(radii) + 1e-12:
                radii = candidate_radii
                orig_pos = candidate
        centers[i] = orig_pos

    return centers, radii

def compute_max_radii(centers):
    """
    Compute the maximum radius for each circle at fixed centers so they don't overlap and stay within [0,1]^2.
    Uses iterative scaling until convergence.
    """
    n = centers.shape[0]
    radii = np.minimum.reduce([centers[:,0], centers[:,1], 1 - centers[:,0], 1 - centers[:,1]])
    radii = np.clip(radii, 0, 1)

    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d <= 1e-15:
                    if radii[i] > 0 or radii[j] > 0:
                        radii[i] = 0.0
                        radii[j] = 0.0
                        changed = True
                else:
                    ri, rj = radii[i], radii[j]
                    if ri + rj > d:
                        scale = d/(ri + rj + 1e-15)
                        new_ri = ri * scale
                        new_rj = rj * scale
                        if abs(new_ri - ri) > 1e-12 or abs(new_rj - rj) > 1e-12:
                            radii[i] = new_ri
                            radii[j] = new_rj
                            changed = True
        if not changed:
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