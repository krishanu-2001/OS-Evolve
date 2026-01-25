# EVOLVE-BLOCK-START
"""Physics‐based circle packing for n=26 circles"""

import numpy as np

def construct_packing():
    """
    Construct and optimize an arrangement of 26 circles in a unit square
    using a force‐based inflate‐and‐relax algorithm to maximize the sum of radii.
    Returns:
        centers: np.array (26,2)
        radii:   np.array (26,)
    """
    np.random.seed(0)
    n = 26

    # 1) Initialize on a coarse grid in [0.1,0.9]^2, then jitter
    gx = np.linspace(0.1, 0.9, 6)
    gy = np.linspace(0.1, 0.9, 5)
    pts = np.array([(x, y) for x in gx for y in gy])
    centers = pts[:n].copy()
    centers += (np.random.rand(n, 2) - 0.5) * 0.02
    centers = np.clip(centers, 0.01, 0.99)

    # 2) Iteratively compute radii and apply force‐based relaxation
    radii = compute_max_radii(centers)
    alpha = 0.02
    for it in range(600):
        forces = np.zeros((n, 2))

        # Pairwise overlap repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dxy = centers[i] - centers[j]
                dist = np.hypot(dxy[0], dxy[1]) + 1e-8
                allow = radii[i] + radii[j]
                if dist < allow:
                    # push them apart
                    overlap = (allow - dist) / dist
                    forces[i] +=  dxy * overlap
                    forces[j] -=  dxy * overlap

        # Border corrective forces
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # left border
            if x - r < 0:
                forces[i, 0] += (r - x)
            # right border
            if x + r > 1:
                forces[i, 0] -= (x + r - 1)
            # bottom border
            if y - r < 0:
                forces[i, 1] += (r - y)
            # top border
            if y + r > 1:
                forces[i, 1] -= (y + r - 1)

        # Update centers
        centers += alpha * forces
        centers = np.clip(centers, 0.01, 0.99)

        # Recompute radii and decay step
        radii = compute_max_radii(centers)
        alpha *= 0.995

    return centers, radii

def compute_max_radii(centers):
    """
    Given fixed centers, compute max radii so circles stay within [0,1]^2 and don't overlap.
    Uses a few relaxation passes to resolve pairwise constraints.
    """
    n = centers.shape[0]
    # initial border‐limited radii
    xs, ys = centers[:,0], centers[:,1]
    radii = np.minimum.reduce([xs, ys, 1 - xs, 1 - ys])

    # Relax pairwise constraints
    for _ in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                dxy = centers[i] - centers[j]
                dist = np.hypot(dxy[0], dxy[1])
                max_sum = radii[i] + radii[j]
                if max_sum > dist:
                    scale = dist / max_sum
                    radii[i] *= scale
                    radii[j] *= scale
    return radii
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii