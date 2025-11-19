# EVOLVE-BLOCK-START
"""
Hybrid gap-driven multi-scale Voronoi initialization + dynamic cluster-aware adaptive annealing
+ enhanced local greedy repacking with constraint-aware force micro-adjustment
for packing 26 circles in a unit square.
"""

import numpy as np
from scipy.spatial import cKDTree, Voronoi

def construct_packing():
    n = 26
    margin = 0.02
    rng = np.random.default_rng(202406)

    # Step 1: Multi-scale gap-driven Voronoi initialization with strategic corner/edge seeding
    centers = multi_scale_gap_init(n, margin, rng)

    # Step 2: Compute initial maximal radii
    radii = compute_max_radii(centers)

    # Step 3: Adaptive simulated annealing with dynamic cluster moves and per-circle adaptive step
    centers, radii = cluster_aware_annealing(centers, radii, margin, rng)

    # Step 4: Enhanced local greedy repacking with constraint-aware force micro-adjustment
    centers, radii = enhanced_local_repacking(centers, radii, margin)

    return centers, radii


def multi_scale_gap_init(n, margin, rng):
    """
    Multi-scale gap-driven initialization:
    - Seed 4 large circles near corners
    - Seed 8 medium circles along edges evenly spaced
    - Fill remaining with iterative Voronoi vertex sampling at multiple scales:
      first large gaps, then smaller gaps, with jitter to break symmetry.
    This hybrid approach better exploits geometric voids and variable circle sizes.
    """
    centers = []

    # Place 4 large circles near corners (indices 0-3)
    corner_offset = 0.06
    corners = np.array([
        [margin + corner_offset, margin + corner_offset],
        [1 - margin - corner_offset, margin + corner_offset],
        [margin + corner_offset, 1 - margin - corner_offset],
        [1 - margin - corner_offset, 1 - margin - corner_offset],
    ])
    centers.extend(corners.tolist())

    # Place 8 medium circles along edges (indices 4-11)
    edge_positions = np.linspace(margin + 0.1, 1 - margin - 0.1, 4)
    # Bottom edge (excluding corners)
    for x in edge_positions[1:-1]:
        centers.append([x, margin + 0.06])
    # Top edge (excluding corners)
    for x in edge_positions[1:-1]:
        centers.append([x, 1 - margin - 0.06])
    # Left edge (excluding corners)
    for y in edge_positions[1:-1]:
        centers.append([margin + 0.06, y])
    # Right edge (excluding corners)
    for y in edge_positions[1:-1]:
        centers.append([1 - margin - 0.06, y])

    centers_arr = np.array(centers)
    remaining = n - len(centers_arr)

    # Multi-scale iterative gap filling using Voronoi vertices
    scales = [0.3, 0.15, 0.08]  # radii to consider for neighbors in cluster moves later
    for scale in scales:
        while remaining > 0:
            # Compute Voronoi with bounding points outside unit square
            bounding_pts = np.array([
                [-1, -1], [-1, 2], [2, -1], [2, 2]
            ])
            all_pts = np.vstack([centers_arr, bounding_pts])
            vor = Voronoi(all_pts)

            # Candidate vertices inside unit square with margin
            candidates = []
            for v in vor.vertices:
                if margin <= v[0] <= 1 - margin and margin <= v[1] <= 1 - margin:
                    candidates.append(v)
            if not candidates:
                break
            candidates = np.array(candidates)

            # Compute max radius at each candidate: min distance to centers and boundary
            tree = cKDTree(centers_arr)
            dists, _ = tree.query(candidates, k=1)
            dist_to_boundary = np.minimum.reduce([
                candidates[:,0] - margin,
                candidates[:,1] - margin,
                (1 - margin) - candidates[:,0],
                (1 - margin) - candidates[:,1]
            ])
            max_radii = np.minimum(dists, dist_to_boundary)

            # Filter candidates with radius at least scale/2 to fill large gaps first
            valid_mask = max_radii >= scale/2
            valid_candidates = candidates[valid_mask]
            valid_radii = max_radii[valid_mask]

            if valid_candidates.shape[0] == 0:
                break

            # Select candidate with largest max radius
            idx = np.argmax(valid_radii)
            new_center = valid_candidates[idx]

            # Add small jitter to break symmetry
            jitter = rng.uniform(-scale*0.05, scale*0.05, size=2)
            new_center = np.clip(new_center + jitter, margin, 1 - margin)

            centers_arr = np.vstack([centers_arr, new_center])
            remaining -= 1

            if remaining == 0:
                break

    # If still remaining, fill randomly inside margin
    if remaining > 0:
        pad = rng.uniform(margin, 1 - margin, size=(remaining, 2))
        centers_arr = np.vstack([centers_arr, pad])

    return centers_arr


def compute_max_radii(centers):
    """
    Iteratively compute maximal non-overlapping radii constrained by neighbors and boundaries.
    """
    n = centers.shape[0]
    radii = np.minimum.reduce([
        centers[:,0] - 0.0,
        centers[:,1] - 0.0,
        1.0 - centers[:,0],
        1.0 - centers[:,1]
    ])

    for _ in range(30):
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


def build_dynamic_clusters(centers, radii, cluster_radius=0.18):
    """
    Build clusters dynamically by grouping circles whose centers lie within cluster_radius.
    Returns list of clusters (lists of indices).
    """
    n = centers.shape[0]
    tree = cKDTree(centers)
    clusters = []
    visited = set()

    for i in range(n):
        if i in visited:
            continue
        neighbors = tree.query_ball_point(centers[i], r=cluster_radius)
        cluster = set(neighbors)
        # Expand cluster by neighbors of neighbors
        to_check = list(cluster)
        while to_check:
            current = to_check.pop()
            if current in visited:
                continue
            visited.add(current)
            new_neighbors = tree.query_ball_point(centers[current], r=cluster_radius)
            for nb in new_neighbors:
                if nb not in cluster:
                    cluster.add(nb)
                    to_check.append(nb)
        clusters.append(sorted(cluster))
    return clusters


def cluster_aware_annealing(centers, radii, margin, rng):
    """
    Adaptive annealing with moves on dynamic clusters and per-circle adaptive step sizes.
    Cluster moves help escape local minima by coordinated shifts.
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

    max_iters = 15000
    stagnation = 0
    max_stagn = 350
    alpha_fast = 0.9945
    alpha_slow = 0.9997

    for it in range(max_iters):
        if stagnation < max_stagn:
            T *= alpha_slow
        else:
            T *= alpha_fast
        if T < T_min:
            T = T_min

        candidate_centers = current_centers.copy()

        # Build dynamic clusters every 100 iterations to reduce overhead
        if it % 100 == 0:
            clusters = build_dynamic_clusters(current_centers, current_radii, cluster_radius=0.18)

        # Choose move type: cluster move or single move
        if rng.uniform() < 0.5 and clusters:
            # Select a random cluster weighted by cluster size (favor larger clusters)
            sizes = np.array([len(c) for c in clusters])
            weights = sizes / sizes.sum()
            cluster_idx = rng.choice(len(clusters), p=weights)
            cluster = clusters[cluster_idx]

            # Move entire cluster by a small random delta scaled by temperature
            step_size = 0.03 * (T / T_init)**0.5
            delta = rng.uniform(-step_size, step_size, size=2)

            for i in cluster:
                candidate_centers[i] += delta
                candidate_centers[i] = np.clip(candidate_centers[i], margin, 1 - margin)
        else:
            # Single circle move with adaptive step size based on radius
            idx = rng.integers(n)
            confinement = current_radii[idx]
            base_step = 0.045 * (T / T_init)**0.5
            step_size = base_step * np.clip(confinement * 10.0, 0.25, 1.0)
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

        if stagnation > 1000:
            T = T_init
            stagnation = 0

    return best_centers, best_radii


def enhanced_local_repacking(centers, radii, margin):
    """
    Local greedy repacking with small directional moves plus a final constraint-aware force micro-adjustment.
    The force phase applies repulsive forces from neighbors and boundaries and small attractive forces to break deadlocks.
    """
    n = centers.shape[0]
    centers = centers.copy()
    radii = radii.copy()
    rng = np.random.default_rng(98765)

    max_local_iters = 4000
    step = 0.008

    # Directional moves to locally improve sum of radii
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
                if new_sum > best_local_sum + 1e-10:
                    best_local_sum = new_sum
                    best_local_center = new_center.copy()

            centers[i] = best_local_center
            if best_local_sum > base_sum + 1e-10:
                improved = True

        if not improved:
            break

    radii = compute_max_radii(centers)

    # Force-based micro-adjustment phase to resolve subtle overlaps and improve packing
    fmax_iter = 2500
    force_step = step / 5
    for _ in range(fmax_iter):
        force_improved = False
        radii = compute_max_radii(centers)
        for i in range(n):
            force = np.zeros(2)
            ci = centers[i]
            ri = radii[i]

            # Repulsive forces from boundaries to maintain margin
            for dim in range(2):
                dist_low = ci[dim] - margin
                dist_high = 1 - margin - ci[dim]
                if dist_low < ri:
                    force[dim] += (ri - dist_low) * 1.2
                if dist_high < ri:
                    force[dim] -= (ri - dist_high) * 1.2

            # Repulsive forces from neighbors
            for j in range(n):
                if j == i:
                    continue
                cj = centers[j]
                rj = radii[j]
                vec = ci - cj
                d = np.linalg.norm(vec) + 1e-14
                overlap = ri + rj - d
                if overlap > 0:
                    force += (vec / d) * overlap * 0.6

            # Small attractive force toward center to prevent drifting too far
            center_bias = np.array([0.5, 0.5]) - ci
            force += center_bias * 0.005

            f_norm = np.linalg.norm(force)
            if f_norm > 1e-10:
                new_ci = ci + (force / f_norm) * min(force_step, f_norm)
                new_ci = np.clip(new_ci, margin, 1 - margin)
                old_radii = radii[i]
                centers[i] = new_ci
                new_radii = compute_max_radii(centers)
                new_sum = np.sum(new_radii)
                old_sum = np.sum(radii)

                # Accept only if sum of radii improves or stays equal (allow plateaus)
                if new_sum + 1e-12 >= old_sum:
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