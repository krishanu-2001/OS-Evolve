# EVOLVE-BLOCK-START
"""
Adaptive clustered simulated annealing with strategic initialization and local greedy repacking
for packing 26 circles in a unit square.
"""

import numpy as np
from scipy.spatial import cKDTree

def construct_packing():
    n = 26
    margin = 0.02
    rng = np.random.default_rng(12345)

    # Enhanced layered initialization with multiple size layers
    centers = layered_init(n, margin)

    # Compute initial radii
    radii = compute_max_radii(centers)

    # Run multiple greedy refinement passes with decreasing step sizes
    centers, radii = multi_pass_greedy(centers, radii, margin)

    # Optional: run annealing for further refinement
    centers, radii = adaptive_simulated_annealing(centers, radii, margin, rng)

    return centers, radii


def layered_init(n, margin):
    """
    Initialize circles in multiple layers:
    - 4 large circles at corners
    - 8 medium circles along edges
    - Multiple small circles in concentric hex layers near center
    """
    centers = np.zeros((n, 2))
    # Place large corners
    corner_offset = 0.07
    corners = np.array([
        [margin + corner_offset, margin + corner_offset],
        [1 - margin - corner_offset, margin + corner_offset],
        [margin + corner_offset, 1 - margin - corner_offset],
        [1 - margin - corner_offset, 1 - margin - corner_offset],
    ])
    centers[0:4] = corners

    # Place medium along edges
    edge_positions = [
        ([corners[0,0]+0.05, corners[0,1]+0.1], [corners[1,0]-0.05, corners[1,1]+0.1]),
        ([corners[0,0]+0.1, corners[0,1]+0.05], [corners[2,0]+0.1, corners[2,1]-0.05]),
        ([corners[2,0]+0.05, corners[2,1]-0.1], [corners[3,0]-0.05, corners[3,1]-0.1]),
        ([corners[1,0]+0.1, corners[1,1]+0.05], [corners[3,0]-0.1, corners[3,1]-0.05])
    ]
    # Assign positions for 8 medium circles
    centers[4:12, 0] = [
        edge_positions[0][0][0], edge_positions[0][1][0],
        edge_positions[1][0][0], edge_positions[1][1][0],
        edge_positions[2][0][0], edge_positions[2][1][0],
        edge_positions[3][0][0], edge_positions[3][1][0]
    ]
    centers[4:12, 1] = [
        edge_positions[0][0][1], edge_positions[0][1][1],
        edge_positions[1][0][1], edge_positions[1][1][1],
        edge_positions[2][0][1], edge_positions[2][1][1],
        edge_positions[3][0][1], edge_positions[3][1][1]
    ]

    # Place small circles in concentric hex layers near center
    layers = 3
    layer_radii = [0.15, 0.2, 0.25]
    count = 12
    for layer_idx, r in enumerate(layer_radii):
        layer_centers = hex_cluster(6*(layer_idx+1), center=(0.5, 0.5), radius=r, margin=margin)
        n_layer = layer_centers.shape[0]
        centers[count:count+n_layer] = layer_centers
        count += n_layer
        if count >= n:
            break
    # If not enough, fill randomly
    if count < n:
        fill = np.random.uniform(margin, 1 - margin, size=(n - count, 2))
        centers[count:] = fill
    return centers


def hex_cluster(num, center, radius, margin):
    """
    Generate a hexagonal cluster of 'num' points centered at 'center'
    inside the unit square with specified 'radius' (approx cluster radius).
    """
    # Compute rows needed to cover num points in hex packing
    # Approximate rows: solve 3*rows^2 - 3*rows +1 >= num
    import math
    rows = 1
    while 3*rows*rows - 3*rows + 1 < num:
        rows += 1

    pts = []
    dx = radius / rows
    dy = dx * np.sqrt(3)/2
    count = 0
    for i in range(rows):
        row_len = rows + i
        y = center[1] - dy*(rows-1)/2 + i*dy
        x_start = center[0] - dx*(row_len-1)/2
        for j in range(row_len):
            if count >= num:
                break
            x = x_start + j*dx
            # Clip inside margins
            x = np.clip(x, margin, 1 - margin)
            yc = np.clip(y, margin, 1 - margin)
            pts.append([x, yc])
            count += 1
        if count >= num:
            break
    pts = np.array(pts)
    # If not enough points, pad randomly inside margin
    if pts.shape[0] < num:
        pad = np.random.uniform(margin, 1 - margin, size=(num - pts.shape[0], 2))
        pts = np.vstack([pts, pad])
    return pts[:num]


def compute_max_radii(centers):
    """
    Compute maximal non-overlapping radii inside unit square by iterative constraint enforcement.
    """
    n = centers.shape[0]
    radii = np.minimum.reduce([
        centers[:,0],            # distance to left
        centers[:,1],            # distance to bottom
        1 - centers[:,0],        # right
        1 - centers[:,1]         # top
    ])

    for _ in range(20):  # more iterations for convergence
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d <= 0:
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


def adaptive_simulated_annealing(centers, radii, margin, rng):
    """
    Perform adaptive simulated annealing with cluster moves and adaptive temperature schedule.
    """
    n = centers.shape[0]
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    current_centers = centers.copy()
    current_radii = radii.copy()
    current_score = best_score

    # KD-tree for neighbor queries
    tree = cKDTree(current_centers)

    T_init = 1e-2
    T_min = 1e-5
    T = T_init

    max_iters = 8000
    stagnation = 0
    max_stagn = 400
    alpha_fast = 0.995
    alpha_slow = 0.9995

    for it in range(max_iters):
        # Adaptive temperature decay
        if stagnation < max_stagn:
            T *= alpha_slow
        else:
            T *= alpha_fast
        if T < T_min:
            T = T_min

        candidate_centers = current_centers.copy()

        # Decide cluster or single move
        if rng.uniform() < 0.40:
            # Cluster move: select a random circle and move it and its neighbors
            idx = rng.integers(n)
            neighbors = tree.query_ball_point(current_centers[idx], r=0.15)
            # limit cluster size to max 4
            if len(neighbors) > 4:
                neighbors = rng.choice(neighbors, size=4, replace=False)
            step_size = 0.02 * (T / T_init)**0.5
            delta = rng.uniform(-step_size, step_size, size=2)
            for i in neighbors:
                candidate_centers[i] += delta
                candidate_centers[i] = np.clip(candidate_centers[i], margin, 1 - margin)
        else:
            # Single circle move
            idx = rng.integers(n)
            step_size = 0.03 * (T / T_init)**0.5
            delta = rng.uniform(-step_size, step_size, size=2)
            candidate_centers[idx] += delta
            candidate_centers[idx] = np.clip(candidate_centers[idx], margin, 1 - margin)

        candidate_radii = compute_max_radii(candidate_centers)
        candidate_score = np.sum(candidate_radii)
        dE = candidate_score - current_score

        if dE > 0 or rng.uniform() < np.exp(dE / T):
            current_centers = candidate_centers
            current_radii = candidate_radii
            current_score = candidate_score
            tree = cKDTree(current_centers)
            if current_score > best_score:
                best_centers = current_centers.copy()
                best_radii = current_radii.copy()
                best_score = current_score
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1

        # Restart temperature if stagnation too long to escape local minima
        if stagnation > 600:
            T = T_init
            stagnation = 0

    return best_centers, best_radii


def multi_pass_greedy(centers, radii, margin):
    """
    Perform multiple greedy passes with decreasing step sizes
    to iteratively improve circle positions.
    """
    n = centers.shape[0]
    centers = centers.copy()
    radii = radii.copy()
    steps = [0.015, 0.008, 0.004]
    for step in steps:
        improved = True
        while improved:
            improved = False
            for i in range(n):
                base_center = centers[i].copy()
                base_radii = compute_max_radii(centers)
                base_sum = np.sum(base_radii)

                directions = np.array([
                    [step,0], [-step,0], [0,step], [0,-step],
                    [step,step], [step,-step], [-step,step], [-step,-step]
                ])

                for d in directions:
                    new_center = base_center + d
                    new_center = np.clip(new_center, margin, 1 - margin)
                    centers[i] = new_center
                    new_radii = compute_max_radii(centers)
                    new_sum = np.sum(new_radii)
                    if new_sum > base_sum + 1e-8:
                        base_sum = new_sum
                        improved = True
                        break
                else:
                    centers[i] = base_center
    return centers, compute_max_radii(centers)

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii