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
    n = 26
    centers = np.zeros((n, 2))

    # Place 4 larger circles near corners (but inset slightly)
    corner_offset = 0.15
    corners = [(corner_offset, corner_offset),
               (corner_offset, 1 - corner_offset),
               (1 - corner_offset, corner_offset),
               (1 - corner_offset, 1 - corner_offset)]
    for i in range(4):
        centers[i] = corners[i]

    # Place 8 medium circles along edges, evenly spaced
    edge_y = [corner_offset, 1 - corner_offset]
    edge_x = [corner_offset, 1 - corner_offset]
    # bottom edge (4 circles)
    for i in range(4):
        centers[4 + i] = [0.25 + 0.15 * i, corner_offset]
    # top edge (4 circles)
    for i in range(4):
        centers[8 + i] = [0.25 + 0.15 * i, 1 - corner_offset]

    # Place 9 circles in a 3x3 grid near center with spacing 0.18
    center_grid_start = 0.32
    spacing = 0.18
    idx = 12
    for row in range(3):
        for col in range(3):
            centers[idx] = [center_grid_start + col * spacing, center_grid_start + row * spacing]
            idx += 1

    # Place 5 small circles in remaining positions near center but offset
    small_positions = [
        (0.5, 0.15),
        (0.5, 0.85),
        (0.15, 0.5),
        (0.85, 0.5),
        (0.5, 0.5)
    ]
    for i in range(5):
        centers[idx] = small_positions[i]
        idx += 1

    # Clip to ensure all centers are inside the unit square with margin
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)
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

    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
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