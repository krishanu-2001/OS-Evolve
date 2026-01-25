# EVOLVE-BLOCK-START
"""Modular physics-based circle packing for n=26 circles with vectorized operations"""

import numpy as np

def initialize_centers(n, seed=0):
    """
    Initialize circle centers on a jittered grid inside [0.1,0.9]^2.
    """
    np.random.seed(seed)
    gx = np.linspace(0.1, 0.9, 6)
    gy = np.linspace(0.1, 0.9, 5)
    pts = np.array([(x, y) for x in gx for y in gy])
    centers = pts[:n].copy()
    jitter = (np.random.rand(n, 2) - 0.5) * 0.02
    centers += jitter
    return np.clip(centers, 0.01, 0.99)

def compute_max_radii(centers, relaxation_iters=5):
    """
    Compute max possible radii so circles fit inside [0,1]^2 and don't overlap.
    Uses iterative relaxation to enforce pairwise constraints.
    """
    n = centers.shape[0]
    xs, ys = centers[:, 0], centers[:, 1]
    radii = np.minimum.reduce([xs, ys, 1 - xs, 1 - ys])

    for _ in range(relaxation_iters):
        # Vectorized pairwise distance matrix
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # shape (n,n,2)
        dist = np.linalg.norm(diff, axis=2) + np.eye(n)  # add eye to avoid zero distance on diagonal
        sum_radii = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap_mask = (sum_radii > dist) & (dist > 0)

        # For each overlapping pair, compute scale factor
        scale = np.ones_like(dist)
        scale[overlap_mask] = dist[overlap_mask] / sum_radii[overlap_mask]

        # For each circle, find minimal scale imposed by any overlap
        min_scale = scale.min(axis=1)
        radii *= min_scale
        # Prevent radii from becoming negative or zero
        radii = np.maximum(radii, 1e-6)
    return radii

def compute_forces(centers, radii):
    """
    Compute repulsion forces between overlapping circles and corrective forces from borders.
    Returns force array of shape (n,2).
    """
    n = centers.shape[0]
    forces = np.zeros_like(centers)

    # Vectorized pairwise differences and distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n,n,2)
    dist = np.linalg.norm(diff, axis=2) + np.eye(n)  # (n,n), add eye to avoid zero division
    sum_radii = radii[:, np.newaxis] + radii[np.newaxis, :]

    overlap = sum_radii - dist
    overlap_mask = (overlap > 0) & (np.eye(n) == 0)  # exclude diagonal

    # Compute normalized direction vectors for overlaps
    direction = np.zeros_like(diff)
    direction[overlap_mask] = diff[overlap_mask] / dist[overlap_mask][:, None]

    # Repulsion force magnitude proportional to overlap
    force_magnitudes = np.zeros((n, n))
    force_magnitudes[overlap_mask] = overlap[overlap_mask] / dist[overlap_mask]

    # Sum forces from all pairs
    forces += np.sum(direction * force_magnitudes[:, :, None], axis=1)
    forces -= np.sum(direction * force_magnitudes[:, :, None], axis=0)

    # Border corrective forces
    x, y = centers[:, 0], centers[:, 1]
    r = radii

    # Left border
    left_overlap = r - x
    forces[:, 0] += np.where(left_overlap > 0, left_overlap, 0)
    # Right border
    right_overlap = (x + r) - 1
    forces[:, 0] -= np.where(right_overlap > 0, right_overlap, 0)
    # Bottom border
    bottom_overlap = r - y
    forces[:, 1] += np.where(bottom_overlap > 0, bottom_overlap, 0)
    # Top border
    top_overlap = (y + r) - 1
    forces[:, 1] -= np.where(top_overlap > 0, top_overlap, 0)

    return forces

def update_centers(centers, forces, alpha, momentum, prev_update):
    """
    Update centers with forces, step size alpha, and momentum.
    """
    update = alpha * forces + momentum * prev_update
    new_centers = centers + update
    new_centers = np.clip(new_centers, 0.01, 0.99)
    return new_centers, update

def construct_packing():
    """
    Construct and optimize arrangement of 26 circles in unit square
    using modular force-based inflate-and-relax algorithm.
    Returns:
        centers: np.array (26,2)
        radii: np.array (26,)
    """
    n = 26
    centers = initialize_centers(n)
    radii = compute_max_radii(centers)

    alpha = 0.025
    momentum = 0.5
    prev_update = np.zeros_like(centers)
    max_iters = 700

    for _ in range(max_iters):
        forces = compute_forces(centers, radii)
        centers, prev_update = update_centers(centers, forces, alpha, momentum, prev_update)
        radii = compute_max_radii(centers)
        alpha *= 0.997  # gradual decay for stability

    return centers, radii

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
