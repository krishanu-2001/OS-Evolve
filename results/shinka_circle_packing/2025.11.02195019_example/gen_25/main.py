# EVOLVE-BLOCK-START
"""Hybrid grid and force-based circle packing for n=26 circles"""

import numpy as np

def construct_packing():
    """
    Initialize circles on a structured grid, then refine via force relaxation.
    Returns:
        centers: np.array (26,2)
        radii: np.array (26,)
        sum_of_radii: float
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Step 1: Structured grid initialization
    # Place 4 large circles near corners
    margin = 0.1
    large_radius = 0.1
    centers[0] = [margin + large_radius, margin + large_radius]
    centers[1] = [1 - margin - large_radius, margin + large_radius]
    centers[2] = [margin + large_radius, 1 - margin - large_radius]
    centers[3] = [1 - margin - large_radius, 1 - margin - large_radius]
    
    # Place 8 medium circles along edges
    edge_positions = np.linspace(margin + 0.2, 1 - margin - 0.2, 4)
    # bottom edge
    for i, x in enumerate(edge_positions):
        centers[4 + i] = [x, margin + 0.05]
    # top edge
    for i, x in enumerate(edge_positions):
        centers[8 + i] = [x, 1 - margin - 0.05]
    # left edge
    for i, y in enumerate(edge_positions):
        centers[12 + i] = [margin + 0.05, y]
    # right edge
    for i, y in enumerate(edge_positions):
        centers[16 + i] = [1 - margin - 0.05, y]
    
    # Place 9 small circles in a 3x3 grid near center
    grid_start = 0.35
    spacing = 0.15
    idx = 20
    for row in range(3):
        for col in range(3):
            centers[idx] = [grid_start + col * spacing, grid_start + row * spacing]
            idx += 1
    
    # Clip to stay within bounds
    centers = np.clip(centers, 0.01, 0.99)
    
    # Step 2: Force-based relaxation
    radii = compute_max_radii(centers)
    alpha = 0.02
    for _ in range(300):
        forces = np.zeros_like(centers)
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d) + 1e-8
                allow = radii[i] + radii[j]
                if dist < allow:
                    overlap = (allow - dist) / dist
                    forces[i] += d * overlap
                    forces[j] -= d * overlap
        # Boundary forces
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Borders
            forces[i, 0] += (r - x) if x - r < 0 else 0
            forces[i, 0] -= (x + r - 1) if x + r > 1 else 0
            forces[i, 1] += (r - y) if y - r < 0 else 0
            forces[i, 1] -= (y + r - 1) if y + r > 1 else 0
        # Update centers
        centers += alpha * forces
        centers = np.clip(centers, 0.01, 0.99)
        radii = compute_max_radii(centers)
        alpha *= 0.995
    return centers, radii

def compute_max_radii(centers):
    """
    Compute maximum radii for circles at given centers without overlap or crossing boundaries.
    """
    n = centers.shape[0]
    radii = np.ones(n)
    # Limit by borders
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    # Limit by overlaps
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d:
                scale = d / (radii[i] + radii[j])
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