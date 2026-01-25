# EVOLVE-BLOCK-START
"""Hybrid hexagonal initialization + hill‐climbing for n=26 circle packing"""

import numpy as np

def construct_packing():
    """
    Initialize 26 circle centers on a hexagonal lattice, then perform
    a hill-climbing local search to maximize the sum of radii.
    Returns:
        centers: np.array (26,2)
        radii:   np.array (26,)
    """
    n = 26
    margin = 0.02
    # Candidate row layouts summing to 26, exploring variations
    candidate_row_layouts = [
        [6, 5, 6, 5, 4],
        [5, 6, 5, 6, 4],
        [6, 6, 5, 5, 4],
        [5, 5, 6, 6, 4],
        [6, 5, 5, 6, 4]
    ]

    best_sum = -1
    best_centers = None
    best_radii = None

    for row_counts in candidate_row_layouts:
        max_cols = max(row_counts)
        dx = (1 - 2*margin) / max_cols
        h  = dx * np.sqrt(3) / 2

        centers = np.zeros((n, 2))
        idx = 0
        for rid, cnt in enumerate(row_counts):
            x_start = margin + (max_cols - cnt) * dx / 2
            y = margin + rid * h
            for c in range(cnt):
                centers[idx, 0] = x_start + c * dx
                centers[idx, 1] = y
                idx += 1

        radii = compute_max_radii(centers)
        s = radii.sum()
        if s > best_sum:
            best_sum = s
            best_centers = centers
            best_radii = radii

    # Hill‐climbing parameters
    iters = 5000
    initial_alpha = dx * 0.5
    rng = np.random.default_rng(42)

    for t in range(iters):
        # decaying step size
        alpha = initial_alpha * (1 - t / iters)
        cand_centers = best_centers.copy()

        # With 20% probability, perturb multiple centers simultaneously
        if rng.uniform() < 0.2:
            count = rng.integers(2, 5)  # perturb 2 to 4 centers
            indices = rng.choice(n, size=count, replace=False)
            for i in indices:
                delta = rng.uniform(-alpha, alpha, size=2)
                cand_centers[i] += delta
                cand_centers[i] = np.clip(cand_centers[i], margin, 1 - margin)
        else:
            i = int(rng.integers(n))
            delta = rng.uniform(-alpha, alpha, size=2)
            cand_centers[i] += delta
            cand_centers[i] = np.clip(cand_centers[i], margin, 1 - margin)

        # recompute radii & evaluate
        cand_radii = compute_max_radii(cand_centers)
        s = cand_radii.sum()
        if s > best_sum:
            best_sum     = s
            best_centers = cand_centers
            best_radii   = cand_radii

    return best_centers, best_radii


def compute_max_radii(centers):
    """
    Given circle centers, compute the maximal non-overlapping radii
    within the unit square by iteratively enforcing border and pairwise constraints.
    """
    n = centers.shape[0]
    radii = np.minimum.reduce([
        centers[:,0],            # distance to left
        centers[:,1],            # distance to bottom
        1 - centers[:,0],        # right
        1 - centers[:,1]         # top
    ])

    # Iterative refinement to maximize radii under constraints
    for _ in range(10):  # fixed number of iterations for convergence
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.hypot(*(centers[i] - centers[j]))
                if d <= 0:
                    # coincident centers — collapse both
                    if radii[i] != 0.0 or radii[j] != 0.0:
                        radii[i] = radii[j] = 0.0
                        changed = True
                else:
                    ri, rj = radii[i], radii[j]
                    if ri + rj > d:
                        scale = d / (ri + rj)
                        new_ri = ri * scale
                        new_rj = rj * scale
                        if new_ri < ri or new_rj < rj:
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