# EVOLVE-BLOCK-START
"""Hybrid structured initialization with adaptive force relaxation for n=26"""

import numpy as np

def construct_packing():
    """
    Construct an initial structured layout of 26 circles in the unit square,
    then refine via adaptive force relaxation and iterative radii maximization.
    Returns:
        centers: np.array (26,2)
        radii:   np.array (26,)
    """
    np.random.seed(42)
    n = 26

    # 1. Structured initial placement: grid + border points + random interior
    centers = np.zeros((n, 2))
    # a) Central cluster: 14 points in a hex grid
    hex_rows, hex_cols = 4, 4
    dx = 0.18
    dy = dx * np.sqrt(3) / 2
    base_x, base_y = 0.26, 0.26
    hex_points = []
    for row in range(hex_rows):
        y = base_y + row * dy
        for col in range(hex_cols):
            x = base_x + col * dx + (row % 2) * (dx / 2)
            if 0.08 < x < 0.92 and 0.08 < y < 0.92:
                hex_points.append((x, y))
    hex_points = hex_points[:14]
    centers[:len(hex_points)] = np.array(hex_points)

    # b) Border points: corners and midpoints
    border_points = [
        [0.13, 0.13], [0.87, 0.13], [0.13, 0.87], [0.87, 0.87],
        [0.5, 0.03], [0.5, 0.97],
        [0.03, 0.5], [0.97, 0.5]
    ]
    centers[len(hex_points):len(hex_points)+8] = np.array(border_points)

    # c) Quarters along edges
    edge_points = [
        [0.18, 0.5], [0.82, 0.5], [0.5, 0.18], [0.5, 0.82]
    ]
    centers[len(hex_points)+8:len(hex_points)+12] = np.array(edge_points)

    # d) Random interior points for remaining
    rng = np.random.default_rng(seed=42)
    interior_points = rng.uniform(low=0.34, high=0.66, size=(n - len(centers), 2))
    centers[len(centers):] = interior_points

    centers = np.clip(centers, 0.01, 0.99)

    # 2. Adaptive force relaxation to improve packing
    radii = compute_max_radii(centers)
    step_size = 0.05
    decay = 0.998
    for _ in range(800):
        forces = np.zeros_like(centers)
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dxy = centers[i] - centers[j]
                dist = np.hypot(dxy[0], dxy[1]) + 1e-8
                max_sum = radii[i] + radii[j]
                if dist < max_sum:
                    overlap = (max_sum - dist) / dist
                    force_vec = dxy * overlap
                    forces[i] += force_vec
                    forces[j] -= force_vec
        # Border forces with adaptive thresholds
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Push away from walls if too close
            if x - r < 0:
                forces[i, 0] += (r - x) * 1.5
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * 1.5
            if y - r < 0:
                forces[i, 1] += (r - y) * 1.5
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * 1.5
            # Slight attraction to center to prevent drift
            forces[i] += 0.005 * (np.array([0.5, 0.5]) - centers[i])
        # Move centers
        centers += step_size * forces
        centers = np.clip(centers, 0.008, 0.992)
        # Recompute radii
        radii_new = compute_max_radii(centers)
        if np.max(np.abs(radii_new - radii)) < 1e-4:
            break
        radii = radii_new
        step_size *= decay

    # 3. Final iterative radii adjustment to maximize sum
    for _ in range(10):
        radii = compute_max_radii(centers)
        radii += 0.0005
        radii = np.minimum(radii, 0.2)
        # Enforce non-overlap by shrinking radii if needed
        for i in range(n):
            for j in range(i + 1, n):
                dxy = centers[i] - centers[j]
                dist = np.hypot(dxy[0], dxy[1]) + 1e-8
                max_sum = radii[i] + radii[j]
                if max_sum > dist:
                    scale = dist / max_sum
                    radii[i] *= scale
                    radii[j] *= scale
        radii = np.clip(radii, 0.01, 0.2)

    return centers, radii

def compute_max_radii(centers):
    """
    Compute maximum radii for each circle given centers, respecting borders and non-overlap.
    """
    n = centers.shape[0]
    xs, ys = centers[:,0], centers[:,1]
    radii = np.minimum.reduce([xs, ys, 1 - xs, 1 - ys])
    # Relax pairwise constraints
    for _ in range(8):
        for i in range(n):
            for j in range(i + 1, n):
                dxy = centers[i] - centers[j]
                dist = np.hypot(dxy[0], dxy[1]) + 1e-8
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
