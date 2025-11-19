# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""

import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # Place circles in a structured pattern
    # This is a simple pattern - evolution will improve this

    # First, place a large circle in the center
    centers[0] = [0.5, 0.5]

    # Place 8 circles around it in a ring
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    # Place 16 more circles in an outer ring
    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    # Additional positioning adjustment to make sure all circles
    # are inside the square and don't overlap
    # Clip to ensure everything is inside the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Post-optimization: simulated annealing with adaptive schedule and local greedy repacking
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    current_centers = centers.copy()
    current_sum = best_sum
    T0 = 0.01
    T = T0
    cooling_rate = 0.995
    no_improve = 0
    np.random.seed(1)
    for it in range(1000):
        # occasional multi-circle moves
        if it % 100 == 0:
            k = np.random.randint(2, 5)  # move 2-4 circles
        else:
            k = np.random.randint(1, 3)  # move 1-2 circles
        idxs = np.random.choice(n, k, replace=False)
        scale = 0.05 * (1 - it / 1000)
        trial = current_centers.copy()
        trial[idxs] += np.random.randn(k, 2) * scale
        trial = np.clip(trial, 0.01, 0.99)
        trial_radii = compute_max_radii(trial)
        s = np.sum(trial_radii)
        dE = s - current_sum
        # Metropolis acceptance
        if dE > 0 or np.random.rand() < np.exp(dE / T):
            current_centers = trial
            current_sum = s
            if s > best_sum:
                best_sum = s
                best_centers = trial.copy()
                best_radii = trial_radii.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        # adaptive temperature adjustment
        if no_improve > 50:
            T *= 1.05
            no_improve = 0
        else:
            T *= cooling_rate
    # Local greedy repacking sweep for fine detail
    for i in range(n):
        orig_pos = best_centers[i].copy()
        for step in [0.01, 0.005, 0.001]:
            for dx, dy in [(step,0),(-step,0),(0,step),(0,-step)]:
                cand_centers = best_centers.copy()
                cand_centers[i] = np.clip(orig_pos + np.array([dx, dy]), 0.01, 0.99)
                cand_radii = compute_max_radii(cand_centers)
                cand_sum = np.sum(cand_radii)
                if cand_sum > best_sum:
                    best_sum = cand_sum
                    best_centers = cand_centers.copy()
                    best_radii = cand_radii.copy()
                    orig_pos = best_centers[i]
        best_centers[i] = orig_pos
    centers, radii = best_centers, best_radii
    return centers, radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles: iterate until convergence
    for _ in range(20):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d:
                    scale = d / (radii[i] + radii[j] + 1e-12)
                    old_i, old_j = radii[i], radii[j]
                    radii[i] *= scale
                    radii[j] *= scale
                    if abs(radii[i] - old_i) > 1e-8 or abs(radii[j] - old_j) > 1e-8:
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