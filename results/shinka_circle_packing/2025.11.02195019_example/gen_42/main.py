# EVOLVE-BLOCK-START
"""Hybrid force relaxation + adaptive annealing + local greedy repacking for n=26 circles"""

import numpy as np

def construct_packing():
    np.random.seed(42)
    n = 26

    # 1) Initialization: hex lattice + random jitter + local perturbations
    m = int(np.ceil(np.sqrt(n / 0.866)))  # hex rows approx
    dx = 0.85 / (m - 1)
    dy = dx * np.sqrt(3) / 2
    hex_pts = []
    for i in range(m):
        for j in range(m):
            x = 0.075 + j * dx + (i % 2) * (dx / 2)
            y = 0.075 + i * dy
            if x <= 0.925 and y <= 0.925:
                hex_pts.append((x, y))
    hex_pts = np.array(hex_pts)
    # Sort by distance to center to pick best n points
    center = np.array([0.5, 0.5])
    d_center = np.linalg.norm(hex_pts - center, axis=1)
    idx = np.argsort(d_center)[:n]
    centers = hex_pts[idx].copy()

    # Add small random jitters and local perturbations
    jitter = (np.random.rand(n, 2) - 0.5) * 0.03
    centers += jitter
    # Local perturbations around initial points to increase diversity
    for _ in range(3):
        for i in range(n):
            if np.random.rand() < 0.3:
                centers[i] += (np.random.rand(2) - 0.5) * 0.01
    centers = np.clip(centers, 0.02, 0.98)

    # 2) Force-based iterative relaxation to adjust centers
    radii = compute_max_radii(centers)
    alpha = 0.04  # larger initial step for better early moves
    for it in range(800):
        forces = np.zeros((n, 2))

        # Pairwise repulsion if overlapping
        for i in range(n):
            for j in range(i + 1, n):
                dxy = centers[i] - centers[j]
                dist = np.hypot(dxy[0], dxy[1]) + 1e-12
                allow = radii[i] + radii[j]
                if dist < allow:
                    overlap = (allow - dist) / dist
                    f = dxy * overlap
                    forces[i] += f
                    forces[j] -= f

        # Border repulsive forces with margin 0.02 to avoid sticking
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            margin = 0.02
            if x - r < margin:
                forces[i, 0] += (margin - (x - r))
            if x + r > 1 - margin:
                forces[i, 0] -= ((x + r) - (1 - margin))
            if y - r < margin:
                forces[i, 1] += (margin - (y - r))
            if y + r > 1 - margin:
                forces[i, 1] -= ((y + r) - (1 - margin))

        # Update centers with adaptive step decay
        centers += alpha * forces
        centers = np.clip(centers, 0.02, 0.98)

        # Recompute radii and decay alpha slowly
        radii = compute_max_radii(centers)
        alpha *= 0.997
        if alpha < 1e-4:
            break

    # 3) Adaptive simulated annealing with multi-circle perturbations
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    T0 = 0.005
    T = T0
    no_improve_iters = 0
    max_no_improve = 150
    np.random.seed(123)

    for k in range(3000):
        # Perturb 1-4 circles simultaneously
        k_perturb = np.random.randint(1, 5)
        idxs = np.random.choice(n, k_perturb, replace=False)
        scale = 0.03 * (1 - k / 3000)
        trial_centers = best_centers.copy()
        trial_centers[idxs] += (np.random.randn(k_perturb, 2) * scale)
        trial_centers = np.clip(trial_centers, 0.02, 0.98)
        trial_radii = compute_max_radii(trial_centers)
        trial_sum = np.sum(trial_radii)
        delta = trial_sum - best_sum

        # Accept with Metropolis criterion
        if delta > 0 or np.random.rand() < np.exp(delta / max(T, 1e-12)):
            best_centers = trial_centers
            best_radii = trial_radii
            best_sum = trial_sum
            no_improve_iters = 0
            # Slow down cooling if improving often
            T *= 0.995
        else:
            no_improve_iters += 1
            # Speed up cooling if stuck
            if no_improve_iters > max_no_improve:
                T *= 0.85
                no_improve_iters = 0

        # Prevent T from going too low or high
        T = np.clip(T, 1e-5, T0)

    centers, radii = best_centers, best_radii

    # 4) Final local greedy repacking sweep: adjust each circle's radius and position slightly
    for sweep in range(3):
        for i in range(n):
            # Try small moves in 4 directions to improve radius sum
            current_center = centers[i].copy()
            current_radius = radii[i]
            best_local_center = current_center.copy()
            best_local_radius = current_radius
            best_local_sum = np.sum(radii)
            steps = [np.array([0.005, 0]), np.array([-0.005, 0]), np.array([0, 0.005]), np.array([0, -0.005])]
            for step in steps:
                trial_center = current_center + step
                trial_center = np.clip(trial_center, 0.02, 0.98)
                trial_centers = centers.copy()
                trial_centers[i] = trial_center
                trial_radii = compute_max_radii(trial_centers)
                trial_sum = np.sum(trial_radii)
                if trial_sum > best_local_sum + 1e-9:
                    best_local_sum = trial_sum
                    best_local_center = trial_center
                    best_local_radius = trial_radii[i]
            centers[i] = best_local_center
            radii[i] = best_local_radius

    return centers, radii


def compute_max_radii(centers):
    n = centers.shape[0]
    xs, ys = centers[:, 0], centers[:, 1]
    # Initial radii limited by borders with margin 0.02
    margin = 0.02
    radii = np.minimum.reduce([xs - margin, ys - margin, 1 - margin - xs, 1 - margin - ys])

    # Iterative pairwise radius relaxation with tighter convergence
    for _ in range(70):
        max_change = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dxy = centers[i] - centers[j]
                dist = np.hypot(dxy[0], dxy[1])
                max_sum = radii[i] + radii[j]
                if max_sum > dist and dist > 1e-14:
                    scale = dist / max_sum
                    old_i, old_j = radii[i], radii[j]
                    radii[i] *= scale
                    radii[j] *= scale
                    max_change = max(max_change, abs(radii[i] - old_i), abs(radii[j] - old_j))
        if max_change < 1e-8:
            break

    # Clamp radii to non-negative values (numerical safety)
    radii = np.maximum(radii, 0.0)
    return radii

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii