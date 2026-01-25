# EVOLVE-BLOCK-START
"""CMA-ES-inspired global optimization for circle packing (n=26)"""

import numpy as np

def construct_packing():
    n = 26
    D = 2 * n

    # 1) Initialize mean at a hexagonal-lattice pattern interior to [0,1]^2
    row_counts = [6, 5, 6, 5, 4]
    margin = 0.1
    max_row = max(row_counts)
    dx = (1 - 2 * margin) / (max_row - 1)
    dy = dx * np.sqrt(3) / 2
    centers0 = []
    for i, cnt in enumerate(row_counts):
        y = margin + i * dy
        x0 = margin + (max_row - cnt) * dx / 2
        for j in range(cnt):
            centers0.append([x0 + j * dx, y])
    centers0 = np.array(centers0[:n])
    mean = centers0.flatten()

    # 2) Evolution Strategy parameters
    pop = 30                  # population size
    mu = pop // 2             # number of elites
    generations = 120
    sigma = 0.1               # initial step size
    # log‐weights for recombination
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)

    rng = np.random.default_rng(123)
    best_f = -np.inf
    best_x = mean.copy()

    # 3) Main ES loop: sample, evaluate, select, recombine, step‐size decay
    for gen in range(generations):
        # Sample candidate solutions
        pop_sols = mean + sigma * rng.standard_normal((pop, D))
        pop_sols = np.clip(pop_sols, 0.0, 1.0)

        fitness = np.zeros(pop)
        for k in range(pop):
            c = pop_sols[k].reshape(n, 2)
            r = compute_max_radii(c)
            fitness[k] = np.sum(r)
            if fitness[k] > best_f:
                best_f = fitness[k]
                best_x = pop_sols[k].copy()

        # Select top‐mu and recombine to update mean
        elite_idx = np.argsort(fitness)[-mu:]
        elite = pop_sols[elite_idx]
        mean = (weights[:, None] * elite).sum(axis=0)

        # Gradually reduce step size
        sigma *= 0.99

    # 4) Return the best candidate found
    centers = best_x.reshape(n, 2)
    radii = compute_max_radii(centers)
    return centers, radii

def compute_max_radii(centers):
    """
    Given circle centers, compute the maximal non-overlapping radii
    within the unit square by iterative constraint enforcement.
    """
    n = centers.shape[0]
    # Start with border‐limited radii
    radii = np.minimum.reduce([
        centers[:, 0], centers[:, 1],
        1 - centers[:, 0], 1 - centers[:, 1]
    ])

    # Iteratively enforce pairwise non‐overlap
    for _ in range(25):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d:
                    scale = d / (radii[i] + radii[j] + 1e-12)
                    old_i, old_j = radii[i], radii[j]
                    radii[i] *= scale
                    radii[j] *= scale
                    if abs(radii[i] - old_i) > 1e-8 or abs(radii[j] - old_j) > 1e-8:
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
