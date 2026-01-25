# EVOLVE-BLOCK-START
"""Hybrid hexagon-edge constructor with adaptive force relaxation for n=26."""

import numpy as np

def construct_packing():
    """
    Construct an arrangement of 26 circles in the unit square,
    using a hybrid approach: a central hex block + adaptive edge filling,
    followed by a physics-based force relaxation to maximize the sum of radii.
    
    Returns:
        Tuple of (centers, radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26,) with radius of each circle
    """
    n = 26
    centers = np.zeros((n, 2))

    # 1. Central "hex block": try the densest 16 arrangement (4x4 hex grid)
    hex_rows = 4
    hex_cols = 4
    dx = 0.18
    dy = dx * np.sqrt(3) / 2
    hex_block = []
    base_x = 0.26
    base_y = 0.26
    for row in range(hex_rows):
        y = base_y + row * dy
        for col in range(hex_cols):
            x = base_x + col * dx + (row % 2) * (dx / 2)
            if 0.08 < x < 0.92 and 0.08 < y < 0.92:
                hex_block.append((x, y))
    # Only use up to 14 centers in the core, saves more for borders
    hex_block = hex_block[:14]
    core_n = len(hex_block)
    centers[:core_n] = np.array(hex_block)

    # 2. Place 8 on the border midpoints and near corners
    border_points = [
        [0.13, 0.13], [0.87, 0.13], [0.13, 0.87], [0.87, 0.87],  # near corners
        [0.5, 0.03], [0.5, 0.97],   # top/bottom edges
        [0.03, 0.5], [0.97, 0.5],   # left/right edges
    ]
    centers[core_n:core_n+8] = np.array(border_points)

    # 3. Place remaining 4 in "intermediate" locations - quarters along edge
    edge_points = [
        [0.18, 0.5], [0.82, 0.5], [0.5, 0.18], [0.5, 0.82]
    ]
    centers[core_n+8:core_n+12] = np.array(edge_points)

    # 4. Place last 4 slightly randomized within central region (to improve)
    rng = np.random.default_rng(seed=42)
    centers[core_n+12:] = rng.uniform(
        low=0.34, high=0.66, size=(n-core_n-12, 2)
    )

    # Enforce all centers are within [0.01, 0.99]
    centers = np.clip(centers, 0.01, 0.99)

    # Physics-based (repulsive) relaxation loop for max 300 iterations
    for relax_iter in range(300):
        moved = False
        for i in range(n):
            d_move = np.zeros(2)
            x_i = centers[i]
            for j in range(n):
                if i == j:
                    continue
                x_j = centers[j]
                # Simple repulsion: If too close, push away.
                dist = np.linalg.norm(x_i - x_j)
                min_sep = 2 * 0.025  # Minimal radius guess: 0.025 per circle
                if dist < min_sep:
                    vec = (x_i - x_j) / (dist + 1e-8)
                    strength = 0.08 * (min_sep - dist)
                    d_move += vec * strength
                elif dist < 0.12:
                    vec = (x_i - x_j) / (dist + 1e-8)
                    strength = 0.01 * (0.12 - dist)
                    d_move += vec * strength
            # Gravity toward center to prevent drift to boundary
            d_move += 0.005 * (np.array([0.5, 0.5]) - x_i)
            # Border "push-in" if close to walls
            wall_dist = np.minimum(x_i, 1 - x_i)
            for dim in [0, 1]:
                if x_i[dim] < 0.06:
                    d_move[dim] += 0.03 * (0.06 - x_i[dim])
                elif x_i[dim] > 0.94:
                    d_move[dim] -= 0.03 * (x_i[dim] - 0.94)
            # Move limited step
            d_move = np.clip(d_move, -0.015, 0.015)
            # Apply
            if np.linalg.norm(d_move) > 1e-5:
                moved = True
                centers[i] += d_move
        centers = np.clip(centers, 0.008, 0.992)
        if not moved and relax_iter > 40:
            break

    # Final: maximize radii with strict border and pairwise constraints
    radii = compute_pairwise_max_radii(centers)
    return centers, radii

def compute_pairwise_max_radii(centers):
    """
    Compute maximal radii for each center with iterative border+pairwise constraints.
    """
    n = centers.shape[0]
    radii = np.ones(n)
    # Border constraints
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1-x, 1-y)
    # Iteratively reduce for overlaps
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist:
                    excess = radii[i] + radii[j] - dist
                    share = 0.5 * excess
                    old_ri, old_rj = radii[i], radii[j]
                    radii[i] -= share
                    radii[j] -= share
                    radii[i] = max(radii[i], 1e-4)
                    radii[j] = max(radii[j], 1e-4)
                    if abs(radii[i] - old_ri) > 1e-7 or abs(radii[j] - old_rj) > 1e-7:
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