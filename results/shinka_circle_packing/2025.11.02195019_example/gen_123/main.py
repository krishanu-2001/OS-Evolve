# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree

def construct_packing():
    """
    Construct and optimize 26-circle packing in unit square.
    Phases:
    1) Strategic hybrid initialization (corners, edges, central hex cluster)
    2) Maximal radii computation by iterative relaxation
    3) Cluster-aware adaptive simulated annealing with multi-circle moves
    4) Local greedy refinement for fine tuning
    Returns:
        centers (26,2), radii (26,)
    """
    n = 26
    margin = 0.02
    rng = np.random.default_rng(1234)

    # Phase 1: Strategic hybrid initialization
    centers = strategic_init(n, margin, rng)

    # Phase 2: Compute initial maximal radii with relaxation
    radii = compute_max_radii(centers)

    # Phase 3: Adaptive cluster-aware simulated annealing
    centers, radii = adaptive_simulated_annealing(centers, radii, margin, rng)

    # Phase 4: Local greedy refinement sweep
    centers, radii = local_greedy_repacking(centers, radii, margin)

    return centers, radii


def strategic_init(n, margin, rng):
    """
    Hybrid initialization:
    - 4 large circles near corners
    - 8 medium circles along edges between corners
    - Remaining 14 in a hexagonal cluster near center
    """
    centers = np.zeros((n, 2))

    # Large circles near corners
    corner_r = 0.11
    corners = np.array([
        [margin + corner_r, margin + corner_r],
        [1 - margin - corner_r, margin + corner_r],
        [margin + corner_r, 1 - margin - corner_r],
        [1 - margin - corner_r, 1 - margin - corner_r],
    ])
    centers[0:4] = corners

    # Medium circles on bottom and top edges (4 each)
    edge_r = 0.07
    xs = np.linspace(corners[0,0] + 0.05, corners[1,0] - 0.05, 4)
    bottom = np.column_stack((xs, np.full_like(xs, margin + edge_r)))
    top = np.column_stack((xs, np.full_like(xs, 1 - margin - edge_r)))
    centers[4:12] = np.vstack([bottom, top])

    # Remaining 14 circles in hex cluster centered at (0.5,0.5)
    remaining = n - 12
    hex_pts = hex_cluster(remaining, center=(0.5,0.5), radius=0.3, margin=margin)
    centers[12:] = hex_pts

    # Add small jitter to break symmetry
    centers += (rng.random((n, 2)) - 0.5) * 0.01
    centers = np.clip(centers, margin, 1 - margin)

    return centers


def hex_cluster(num, center, radius, margin):
    """
    Generate approximately 'num' points in a hexagonal cluster near 'center'.
    """
    rows = 1
    while 3*rows*rows - 3*rows + 1 < num:
        rows += 1

    pts = []
    dx = radius / rows
    dy = dx * np.sqrt(3) / 2
    count = 0
    for i in range(rows):
        row_len = rows + i
        y = center[1] - dy*(rows-1)/2 + i*dy
        x_start = center[0] - dx*(row_len-1)/2
        for j in range(row_len):
            if count >= num:
                break
            x = x_start + j*dx
            x = np.clip(x, margin, 1 - margin)
            y_clipped = np.clip(y, margin, 1 - margin)
            pts.append([x, y_clipped])
            count += 1
        if count >= num:
            break
    pts = np.array(pts)
    if pts.shape[0] < num:
        extra = np.random.uniform(margin, 1 - margin, size=(num - pts.shape[0], 2))
        pts = np.vstack([pts, extra])
    return pts[:num]


def compute_max_radii(centers):
    """
    Compute maximal radii ensuring no overlap and boundary containment.
    Iteratively relax pairwise constraints until convergence.
    """
    n = centers.shape[0]
    xs, ys = centers[:,0], centers[:,1]
    radii = np.minimum.reduce([xs, ys, 1 - xs, 1 - ys])

    for _ in range(50):
        max_change = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dxy = centers[i] - centers[j]
                dist = np.hypot(*dxy)
                total = radii[i] + radii[j]
                if total > dist and dist > 1e-12:
                    scale = dist / total
                    old_i, old_j = radii[i], radii[j]
                    radii[i] *= scale
                    radii[j] *= scale
                    max_change = max(max_change, abs(radii[i]-old_i), abs(radii[j]-old_j))
        if max_change < 1e-6:
            break
    return radii


def adaptive_simulated_annealing(centers, radii, margin, rng):
    """
    Simulated annealing with cluster moves using KD-tree neighbor queries.
    Adaptive temperature schedule with escape from stagnation.
    """
    n = centers.shape[0]
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)

    current_centers = centers.copy()
    current_radii = radii.copy()
    current_sum = best_sum

    T_init = 0.01
    T_min = 1e-5
    T = T_init

    stagnation = 0
    max_stagn = 300
    alpha_slow = 0.9995
    alpha_fast = 0.995

    tree = cKDTree(current_centers)

    for it in range(8000):
        if rng.random() < 0.25:
            # Cluster move: pick random circle and neighbors within radius
            idx = rng.integers(n)
            neighbors = tree.query_ball_point(current_centers[idx], r=0.15)
            if len(neighbors) > 4:
                neighbors = rng.choice(neighbors, size=4, replace=False)
            step_size = 0.02 * (T / T_init)**0.5
            delta = rng.uniform(-step_size, step_size, size=2)
            candidate_centers = current_centers.copy()
            for i in neighbors:
                candidate_centers[i] += delta
            candidate_centers = np.clip(candidate_centers, margin, 1 - margin)
        else:
            # Single circle move
            idx = rng.integers(n)
            step_size = 0.03 * (T / T_init)**0.5
            delta = rng.uniform(-step_size, step_size, size=2)
            candidate_centers = current_centers.copy()
            candidate_centers[idx] += delta
            candidate_centers = np.clip(candidate_centers, margin, 1 - margin)

        candidate_radii = compute_max_radii(candidate_centers)
        candidate_sum = np.sum(candidate_radii)
        delta_score = candidate_sum - current_sum

        accept = False
        if delta_score > 0:
            accept = True
        else:
            accept = rng.random() < np.exp(delta_score / T)

        if accept:
            current_centers = candidate_centers
            current_radii = candidate_radii
            current_sum = candidate_sum
            tree = cKDTree(current_centers)
            if current_sum > best_sum + 1e-8:
                best_sum = current_sum
                best_centers = current_centers.copy()
                best_radii = current_radii.copy()
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1

        # Adaptive temperature decay
        if stagnation < max_stagn:
            T *= alpha_slow
        else:
            T *= alpha_fast

        if T < T_min:
            T = T_min

        # Reset temperature if stagnation too long
        if stagnation > 1000:
            T = T_init
            stagnation = 0

    return best_centers, best_radii


def local_greedy_repacking(centers, radii, margin):
    """
    Local greedy refinement: for each circle, try small moves in 8 directions
    to increase sum of radii until no improvement.
    """
    n = centers.shape[0]
    centers = centers.copy()
    radii = radii.copy()
    rng = np.random.default_rng(9999)
    directions = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]])
    step_sizes = [0.01, 0.005, 0.0025, 0.001]

    for step in step_sizes:
        improved = True
        while improved:
            improved = False
            for i in range(n):
                base_center = centers[i].copy()
                base_radii = compute_max_radii(centers)
                base_sum = np.sum(base_radii)

                best_center = base_center.copy()
                best_sum_local = base_sum

                for d in directions:
                    candidate = base_center + d * step
                    candidate = np.clip(candidate, margin, 1 - margin)
                    centers[i] = candidate
                    candidate_radii = compute_max_radii(centers)
                    candidate_sum = np.sum(candidate_radii)
                    if candidate_sum > best_sum_local + 1e-8:
                        best_sum_local = candidate_sum
                        best_center = candidate.copy()

                centers[i] = best_center
                if best_sum_local > base_sum + 1e-8:
                    improved = True

    radii = compute_max_radii(centers)
    return centers, radii

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii