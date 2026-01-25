# EVOLVE-BLOCK-START
"""Hybrid physics-based + adaptive SA circle packing for n=26"""

import numpy as np

def construct_packing():
    """
    Construct and optimize an arrangement of 26 circles in a unit square
    using strategic placement, force-based relaxation,
    adaptive simulated annealing with multi-circle moves,
    and local greedy refinement.
    Returns:
        centers: np.array (26,2)
        radii:   np.array (26,)
    """
    np.random.seed(42)
    n = 26

    # 1) Place larger circles at corners and edges
    centers = []
    # corners
    corner_positions = np.array([
        [0.05, 0.05],
        [0.05, 0.95],
        [0.95, 0.05],
        [0.95, 0.95]
    ])
    centers.extend(corner_positions)
    # midpoints of edges
    edge_positions = np.array([
        [0.5, 0.05],
        [0.5, 0.95],
        [0.05, 0.5],
        [0.95, 0.5]
    ])
    centers.extend(edge_positions)
    # Remaining circles in hexagonal lattice inside [0.15,0.85]^2
    remaining = n - len(centers)
    m = int(np.ceil(np.sqrt(remaining / 0.866)))
    dx = 0.7 / (m - 1)
    dy = dx * np.sqrt(3) / 2
    pts = []
    for i in range(m):
        for j in range(m):
            x = 0.15 + j * dx + (i % 2) * (dx / 2)
            y = 0.15 + i * dy
            if x <= 0.85 and y <= 0.85:
                pts.append((x, y))
    pts = np.array(pts)
    # Select top remaining by distance to border
    d_border = np.min(np.stack([pts, 1 - pts], axis=2), axis=2).min(axis=1)
    idx = np.argsort(-d_border)[:remaining]
    centers.extend(pts[idx])
    centers = np.array(centers)
    # Jitter to break symmetry
    centers += (np.random.rand(n, 2) - 0.5) * 0.02
    centers = np.clip(centers, 0.01, 0.99)

    # 2) Force-based relaxation
    alpha = 0.03
    for it in range(400):
        forces = np.zeros((n, 2))
        # Pairwise repulsion
        for i in range(n):
            for j in range(i+1, n):
                dxy = centers[i] - centers[j]
                dist = np.hypot(*dxy) + 1e-8
                allow = compute_max_radii_pair(centers, i, j)
                if dist < allow:
                    overlap = (allow - dist) / dist
                    forces[i] += dxy * overlap
                    forces[j] -= dxy * overlap
        # Border corrective forces
        for i in range(n):
            x, y = centers[i]
            r = compute_max_radii_single(centers, i)
            if x - r < 0: forces[i,0] += r - x
            if x + r > 1: forces[i,0] -= x + r - 1
            if y - r < 0: forces[i,1] += r - y
            if y + r > 1: forces[i,1] -= y + r - 1
        # Update positions
        centers += alpha * forces
        centers = np.clip(centers, 0.01, 0.99)
        # Recompute radii after move
        radii = compute_max_radii(centers)
        alpha *= 0.995

    # 3) Adaptive simulated annealing
    centers_sa = centers.copy()
    best_centers = centers.copy()
    radii_sa = radii.copy()
    best_radii = radii.copy()
    sum_sa = np.sum(radii_sa)
    best_sum = sum_sa
    T = 0.002
    cooling_rate = 0.995
    max_T = 0.01
    no_improve = 0

    for k in range(1500):
        # multi-circle moves every 100 iters, else single-circle
        if k % 100 == 0:
            count = np.random.randint(2, 5)
            idxs = np.random.choice(n, count, replace=False)
        else:
            idxs = [np.random.randint(n)]
        old_pos = centers_sa[idxs].copy()
        scale = 0.006 * (1 - k / 1500)
        delta = np.random.randn(len(idxs),2) * scale
        centers_sa[idxs] = np.clip(centers_sa[idxs] + delta, 0.01, 0.99)
        radii_tmp = compute_max_radii(centers_sa)
        sum_new = np.sum(radii_tmp)
        dE = sum_new - sum_sa
        # Metropolis acceptance
        if dE > 0 or np.random.rand() < np.exp(dE / T):
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
        # Adaptive cooling
        if no_improve > 100:
            T = min(T * 1.1, max_T)
            no_improve = 0
        else:
            T *= cooling_rate

    centers, radii = best_centers.copy(), best_radii.copy()

    # 4) Local greedy refinement
    directions = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]])
    for i in range(n):
        orig = centers[i].copy()
        for step in [0.01, 0.005, 0.0025, 0.001]:
            improved = True
            while improved:
                improved = False
                for d in directions:
                    cand = orig + d * step
                    cand = np.clip(cand, 0.01, 0.99)
                    centers[i] = cand
                    radii_cand = compute_max_radii(centers)
                    if np.sum(radii_cand) > np.sum(radii):
                        radii = radii_cand
                        orig = cand
                        improved = True
                centers[i] = orig

    return centers, radii

def compute_max_radii(centers):
    """
    Given fixed centers, compute maximal radii within [0,1]^2 without overlap.
    Uses iterative pairwise scaling until convergence.
    """
    n = centers.shape[0]
    xs, ys = centers[:,0], centers[:,1]
    radii = np.minimum.reduce([xs, ys, 1-xs, 1-ys])
    for _ in range(50):
        max_change = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dxy = centers[i] - centers[j]
                dist = np.hypot(*dxy)
                total = radii[i] + radii[j]
                if total > dist and dist > 1e-12:
                    scale = dist / total
                    old_i, old_j = radii[i], radii[j]
                    radii[i] *= scale
                    radii[j] *= scale
                    max_change = max(max_change, abs(radii[i]-old_i), abs(radii[j]-old_j))
        if max_change < 1e-6:
            break
    return radii

def compute_max_radii_pair(centers, i, j):
    """
    Helper to compute sum of radii limits for a pair (i,j) based on border distances.
    """
    r_i = compute_max_radii_single(centers, i)
    r_j = compute_max_radii_single(centers, j)
    return r_i + r_j

def compute_max_radii_single(centers, i):
    """
    Helper to compute max radius of circle i based on border distances.
    """
    x, y = centers[i]
    return min(x, y, 1 - x, 1 - y)
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii