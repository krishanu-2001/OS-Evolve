# EVOLVE-BLOCK-START
"""Repulsive-gradient-based circle packing for n=26 circles"""

import numpy as np

def construct_packing():
    """
    Optimize the positions and radii of 26 circles in a unit square
    to maximize the sum of radii using repulsive-force-based gradient ascent.
    Returns:
        centers: np.array (26,2)
        radii: np.array (26,)
    """
    n = 26
    np.random.seed(0)
    centers = np.random.rand(n, 2) * 0.8 + 0.1  # Initial: not too close to border
    radii = np.full(n, 0.04)  # Reasonable initial guess

    lr_c = 0.015  # Learning rate for centers
    lr_r = 0.01   # Learning rate for radii
    n_steps = 1200
    min_r, max_r = 0.01, 0.5

    for step in range(n_steps):
        grad_c = np.zeros_like(centers)
        grad_r = np.ones_like(radii)  # reward growing each radius

        # Repulsive force between all pairs (overlaps -> penalize, otherwise weak)
        for i in range(n):
            for j in range(i+1, n):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                min_dist = radii[i] + radii[j] + 1e-7
                # If overlaps, strong penalty; otherwise, weak (almost none)
                if dist < min_dist:
                    # Overlap, push away and penalize big radii
                    overlap = min_dist - dist
                    # d direction, if two centers coincide, random
                    if dist < 1e-8:
                        direction = np.random.randn(2)
                        direction /= (np.linalg.norm(direction) + 1e-10)
                    else:
                        direction = d/dist
                    force = 10 * overlap
                    grad_c[i] +=  force * direction
                    grad_c[j] -=  force * direction
                    grad_r[i]   -= 10 * overlap
                    grad_r[j]   -= 10 * overlap

        # Boundary penalty: project centers out and shrink radii on overflow
        for i in range(n):
            x, y, r = centers[i,0], centers[i,1], radii[i]
            # Left, right, bottom, top
            over_left = max(0, r - x)
            over_right = max(0, r - (1 - x))
            over_bottom = max(0, r - y)
            over_top = max(0, r - (1 - y))
            # Push inside
            grad_c[i,0] +=  15 * over_left - 15 * over_right
            grad_c[i,1] +=  15 * over_bottom - 15 * over_top
            grad_r[i]   -= 10 * (over_left + over_right + over_bottom + over_top)

        # Apply update (gradient * learning rate)
        centers += lr_c * grad_c
        radii   += lr_r * grad_r

        # Project radii to [min_r, max_r], centers to [0, 1]
        radii = np.clip(radii, min_r, max_r)
        centers = np.clip(centers, 0.0, 1.0)

    # Final projection: make sure no overlaps & within box
    radii = compute_max_radii_project(centers, radii)
    return centers, radii

def compute_max_radii_project(centers, radii_in):
    """
    Final projection: for each circle, shrink radii to avoid overlaps and boundary violations.
    Args:
        centers: (n,2)
        radii_in: (n,)
    Returns:
        radii: (n,)
    """
    n = len(radii_in)
    radii = np.copy(radii_in)
    # Border constraint
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, 1-x, y, 1-y)
    # Pairwise
    # Do this iteratively so all pairs are checked until stable
    for _ in range(5):
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < radii[i] + radii[j]:
                    # Shrink both to "touch"
                    excess = (radii[i] + radii[j] - d) / 2.0 + 1e-8
                    if radii[i] >= radii[j]:
                        radii[i] -= excess
                    else:
                        radii[j] -= excess
                    radii[i] = max(radii[i], 0.005)
                    radii[j] = max(radii[j], 0.005)
        # border again in each loop
        for i in range(n):
            x, y = centers[i]
            radii[i] = min(radii[i], x, 1-x, y, 1-y)
    return radii

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
