# EVOLVE-BLOCK-START
"""
Gap-driven Voronoi-based initialization + adaptive clustered simulated annealing + local greedy repacking
for packing 26 circles in a unit square.
"""

import numpy as np
from scipy.spatial import cKDTree, Voronoi

def construct_packing():
    n = 26
    margin = 0.02
    rng = np.random.default_rng(12345)

    # Step 1: Gap-driven initialization using Voronoi tessellation
    centers = gap_driven_init(n, margin, rng)

    # Step 2: Compute initial maximal radii
    radii = compute_max_radii(centers)

    # Step 3: Adaptive simulated annealing with cluster moves focused on dense regions
    centers, radii = adaptive_simulated_annealing(centers, radii, margin, rng)

    # Step 4: Local greedy repacking for fine tuning
    centers, radii = local_greedy_repacking(centers, radii, margin)

    return centers, radii


def gap_driven_init(n, margin, rng):
    """
    Hybrid targeted gap sampling and Voronoi-based initialization:
    - Place circles in largest gaps identified by Voronoi vertices,
      and also sample targeted voids (midpoints of distant pairs, underpopulated edge/center regions).
    - Periodically refresh Voronoi diagram and candidate set.
    This approach aims to maximize initial coverage and exploit both global and local geometric voids.
    """
    centers = []
    # Start with 4 large circles near corners to reduce edge effects
    corner_offset = 0.06
    corners = np.array([
        [margin + corner_offset, margin + corner_offset],
        [1 - margin - corner_offset, margin + corner_offset],
        [margin + corner_offset, 1 - margin - corner_offset],
        [1 - margin - corner_offset, 1 - margin - corner_offset],
    ])
    centers.extend(corners.tolist())

    # Add 6 medium circles along edges evenly spaced (3 per side, bottom and left)
    edge_positions = np.linspace(margin + 0.1, 1 - margin - 0.1, 3)
    # Bottom edge (excluding corners)
    for x in edge_positions:
        centers.append([x, margin + 0.06])
    # Left edge (excluding corners)
    for y in edge_positions:
        centers.append([margin + 0.06, y])

    # Now we have 4 + 6 = 10 circles placed, need to place 16 more inside

    centers_arr = np.array(centers)

    for idx_place in range(n - len(centers)):
        # --- Voronoi-based candidates ---
        bounding_pts = np.array([
            [-1, -1], [-1, 2], [2, -1], [2, 2]
        ])
        all_pts = np.vstack([centers_arr, bounding_pts])
        vor = Voronoi(all_pts)

        # Voronoi vertices inside the unit square with margin
        vor_candidates = []
        for v in vor.vertices:
            if margin <= v[0] <= 1 - margin and margin <= v[1] <= 1 - margin:
                vor_candidates.append(v)
        vor_candidates = np.array(vor_candidates) if len(vor_candidates) > 0 else np.zeros((0,2))

        # --- Targeted gap candidates ---
        # 1. Midpoints between most distant pairs (avoid those too close to existing centers)
        tree = cKDTree(centers_arr)
        candidate_midpoints = []
        if len(centers_arr) > 2:
            # Find the 4 most distant pairs
            dmat = np.linalg.norm(centers_arr[None,:,:] - centers_arr[:,None,:], axis=2)
            inds = np.dstack(np.unravel_index(np.argsort(-dmat.ravel()), dmat.shape))[0]
            midpoints = []
            count = 0
            for i,j in inds:
                if i >= j:
                    continue
                midpoint = (centers_arr[i] + centers_arr[j]) / 2
                if margin <= midpoint[0] <= 1 - margin and margin <= midpoint[1] <= 1 - margin:
                    # Only add if not too close to existing
                    min_dist, _ = tree.query(midpoint, k=1)
                    if min_dist > 0.12:
                        midpoints.append(midpoint)
                        count += 1
                if count >= 4:
                    break
            if midpoints:
                candidate_midpoints = np.array(midpoints)
            else:
                candidate_midpoints = np.zeros((0,2))
        else:
            candidate_midpoints = np.zeros((0,2))

        # 2. Edge and center sampling if underpopulated
        # Sample at square center if no circle within 0.15 of center
        center_pt = np.array([[0.5, 0.5]])
        min_dist_center, _ = tree.query(center_pt, k=1)
        center_candidate = center_pt if min_dist_center[0] > 0.15 else np.zeros((0,2))
        # Sample at midpoints of each edge if underpopulated
        edge_pts = np.array([
            [0.5, margin + 0.06],
            [0.5, 1 - margin - 0.06],
            [margin + 0.06, 0.5],
            [1 - margin - 0.06, 0.5]
        ])
        edge_candidates = []
        for pt in edge_pts:
            min_dist, _ = tree.query(pt, k=1)
            if min_dist > 0.13:
                edge_candidates.append(pt)
        edge_candidates = np.array(edge_candidates) if edge_candidates else np.zeros((0,2))

        # Combine all candidates
        candidates = np.vstack([
            vor_candidates,
            candidate_midpoints,
            center_candidate,
            edge_candidates
        ]) if vor_candidates.shape[0] + candidate_midpoints.shape[0] + center_candidate.shape[0] + edge_candidates.shape[0] > 0 else np.zeros((0,2))

        # If no candidates, place random point inside margin
        if candidates.shape[0] == 0:
            new_center = rng.uniform(margin, 1 - margin, size=2)
            centers_arr = np.vstack([centers_arr, new_center])
            continue

        # For each candidate, compute minimal distance to existing centers and to boundary
        dists, _ = tree.query(candidates, k=1)
        dist_to_boundary = np.minimum.reduce([
            candidates[:,0] - margin,
            candidates[:,1] - margin,
            (1 - margin) - candidates[:,0],
            (1 - margin) - candidates[:,1]
        ])
        max_radii = np.minimum(dists, dist_to_boundary)

        # Select candidate with largest max radius
        best_idx = np.argmax(max_radii)
        new_center = candidates[best_idx]

        # Add small random jitter to avoid perfect symmetry
        jitter = rng.uniform(-0.005, 0.005, size=2)
        new_center = np.clip(new_center + jitter, margin, 1 - margin)

        centers_arr = np.vstack([centers_arr, new_center])

        # After every 4 placements, refresh Voronoi diagram and candidate set (implicitly done by loop)

    return centers_arr


def compute_max_radii(centers):
    """
    Compute maximal non-overlapping radii inside unit square by iterative constraint enforcement.
    """
    n = centers.shape[0]
    radii = np.minimum.reduce([
        centers[:,0] - 0.0,            # distance to left
        centers[:,1] - 0.0,            # distance to bottom
        1.0 - centers[:,0],            # right
        1.0 - centers[:,1]             # top
    ])

    for _ in range(30):  # more iterations for convergence
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
    Perform adaptive simulated annealing with cluster moves focused on dense regions and adaptive temperature schedule.
    """
    n = centers.shape[0]
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    current_centers = centers.copy()
    current_radii = radii.copy()
    current_score = best_score

    tree = cKDTree(current_centers)

    T_init = 1e-2
    T_min = 1e-5
    T = T_init

    max_iters = 12000
    stagnation = 0
    max_stagn = 400
    alpha_fast = 0.995
    alpha_slow = 0.9995

    for it in range(max_iters):
        if stagnation < max_stagn:
            T *= alpha_slow
        else:
            T *= alpha_fast
        if T < T_min:
            T = T_min

        candidate_centers = current_centers.copy()

        # Cluster moves focused on densest regions: pick circle with smallest radius and move cluster
        if rng.uniform() < 0.45:
            idx = np.argmin(current_radii)
            neighbors = tree.query_ball_point(current_centers[idx], r=0.15)
            if len(neighbors) > 5:
                neighbors = rng.choice(neighbors, size=5, replace=False)
            step_size = 0.025 * (T / T_init)**0.5
            delta = rng.uniform(-step_size, step_size, size=2)
            for i in neighbors:
                candidate_centers[i] += delta
                candidate_centers[i] = np.clip(candidate_centers[i], margin, 1 - margin)
        else:
            # Single circle move with adaptive step size based on radius
            idx = rng.integers(n)
            confinement = current_radii[idx]
            base_step = 0.04 * (T / T_init)**0.5
            step_size = base_step * np.clip(confinement * 10.0, 0.2, 1.0)
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

        if stagnation > 900:
            T = T_init
            stagnation = 0

    return best_centers, best_radii


def local_greedy_repacking(centers, radii, margin):
    """
    For each circle, locally optimize its position by small moves to increase sum of radii,
    then perform a constraint-aware force-based micro-adjustment phase to fine-tune.
    """
    n = centers.shape[0]
    centers = centers.copy()
    radii = radii.copy()
    rng = np.random.default_rng(98765)

    max_local_iters = 3000
    step = 0.01

    for _ in range(max_local_iters):
        improved = False
        for i in range(n):
            base_center = centers[i].copy()
            base_radii = compute_max_radii(centers)
            base_sum = np.sum(base_radii)

            directions = np.array([
                [step,0], [-step,0], [0,step], [0,-step],
                [step,step], [step,-step], [-step,step], [-step,-step]
            ])

            best_local_center = base_center.copy()
            best_local_sum = base_sum

            for d in directions:
                new_center = base_center + d
                new_center = np.clip(new_center, margin, 1 - margin)
                centers[i] = new_center
                new_radii = compute_max_radii(centers)
                new_sum = np.sum(new_radii)
                if new_sum > best_local_sum:
                    best_local_sum = new_sum
                    best_local_center = new_center.copy()

            centers[i] = best_local_center
            if best_local_sum > base_sum + 1e-8:
                improved = True

        if not improved:
            break

    radii = compute_max_radii(centers)

    # Micro-adjustment phase for subtle overlap relief and final optimization
    fmax_iter = 1500
    force_step = step / 4
    for _ in range(fmax_iter):
        force_improved = False
        radii = compute_max_radii(centers)
        for i in range(n):
            force = np.zeros(2)
            ci = centers[i]
            ri = radii[i]

            # Repulsive forces from borders to keep inside with margin
            for dim in range(2):
                dist_to_low = ci[dim] - margin
                dist_to_high = 1 - margin - ci[dim]
                if dist_to_low < ri:
                    force[dim] += (ri - dist_to_low)
                if dist_to_high < ri:
                    force[dim] -= (ri - dist_to_high)

            # Repulsive forces from neighbors
            for j in range(n):
                if j == i:
                    continue
                cj = centers[j]
                rj = radii[j]
                vec = ci - cj
                d = np.linalg.norm(vec)+1e-12
                overlap = ri + rj - d
                if overlap > 0:
                    force += (vec / d) * overlap * 0.5

            if np.linalg.norm(force) > 1e-8:
                new_ci = ci + force * force_step
                new_ci = np.clip(new_ci, margin, 1 - margin)
                old_radii = radii[i]
                centers[i] = new_ci
                new_radii = compute_max_radii(centers)
                new_sum = np.sum(new_radii)
                old_sum = np.sum(radii)

                if new_sum > old_sum:
                    radii = new_radii
                    force_improved = True
                else:
                    centers[i] = ci

        if not force_improved:
            break

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