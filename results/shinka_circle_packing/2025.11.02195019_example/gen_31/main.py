# EVOLVE-BLOCK-START
import numpy as np

def layout_candidates():
    # Candidate row splits summing to 26 with 4-6 rows, max 7 per row
    return [
        [6,5,6,5,4],
        [5,5,6,5,5],
        [5,6,5,6,4],
        [4,6,6,5,5],
        [6,6,5,5,4],
        [5,6,6,4,5]
    ]

def hex_row_centers(count, y, x_margin=0.05, width=0.9):
    # Generate centers for a hex row with count circles at height y
    # Equally spaced horizontally with small jitter
    if count == 1:
        return np.array([[0.5, y]])
    spacing = (width - 2*x_margin) / (count - 1) if count > 1 else 0
    xs = x_margin + np.arange(count) * spacing
    centers = np.stack([xs, np.full(count, y)], axis=1)
    jitter = (np.random.rand(count) - 0.5) * 0.014
    centers[:,0] += jitter
    return centers

def assign_size_gradient(centers):
    # Assign size factors favoring larger circles near center (0.5,0.5)
    dist_center = np.linalg.norm(centers - 0.5, axis=1)
    max_dist = np.sqrt(2*0.5**2)
    norm_dist = dist_center / max_dist
    size_factor = np.exp(- (norm_dist*4)**2)
    size_factor = 0.1 + 0.9 * (size_factor - size_factor.min())/(size_factor.max() - size_factor.min() + 1e-10)
    return size_factor

def compute_max_radii(centers, max_iters=10):
    n = centers.shape[0]
    radii = np.minimum.reduce([centers[:,0], centers[:,1], 1 - centers[:,0], 1 - centers[:,1]])
    for _ in range(max_iters):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i]-centers[j])
                if d <= 1e-14:
                    continue
                if radii[i] + radii[j] > d:
                    scale = d / (radii[i] + radii[j])
                    if scale < 1.0 - 1e-8:
                        radii[i] *= scale
                        radii[j] *= scale
                        changed = True
        if not changed:
            break
    radii = np.maximum(radii, 1e-6)
    return radii

def compute_forces(centers, radii):
    n = centers.shape[0]
    forces = np.zeros_like(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n,n,2)
    dist = np.linalg.norm(diff, axis=2) + np.eye(n)
    sum_r = radii[:, None] + radii[None, :]
    overlap = sum_r - dist
    overlap_mask = (overlap > 0) & (~np.eye(n, dtype=bool))

    direction = np.zeros_like(diff)
    direction[overlap_mask] = diff[overlap_mask] / dist[overlap_mask][:, None]

    # Use squared overlap for stronger repulsion
    strength = np.zeros_like(dist)
    strength[overlap_mask] = (overlap[overlap_mask] ** 2) / dist[overlap_mask]

    forces += np.sum(direction * strength[:, :, None], axis=1)
    forces -= np.sum(direction * strength[:, :, None], axis=0)

    x, y = centers[:,0], centers[:,1]
    r = radii

    overlap_l = r - x
    forces[:,0] += np.where(overlap_l > 0, overlap_l**2, 0)
    overlap_r = (x + r) - 1
    forces[:,0] -= np.where(overlap_r > 0, overlap_r**2, 0)
    overlap_b = r - y
    forces[:,1] += np.where(overlap_b > 0, overlap_b**2, 0)
    overlap_t = (y + r) - 1
    forces[:,1] -= np.where(overlap_t > 0, overlap_t**2, 0)

    return forces

def update_centers(centers, forces, alpha, momentum, prev_update):
    update = alpha * forces + momentum * prev_update
    centers_new = centers + update
    centers_new = np.clip(centers_new, 0.01, 0.99)
    return centers_new, update

def construct_packing():
    np.random.seed(42)
    n = 26
    best_centers = None
    best_radii = None
    best_sum = -np.inf

    for rows in layout_candidates():
        if sum(rows) != n:
            continue
        nrows = len(rows)
        y_margin = 0.07
        y_positions = np.linspace(y_margin, 1 - y_margin, nrows)

        centers_list = []
        for i, count in enumerate(rows):
            row_centers = hex_row_centers(count, y_positions[i], x_margin=0.05, width=0.9)
            centers_list.append(row_centers)
        centers = np.vstack(centers_list)

        size_factor = assign_size_gradient(centers)
        base_radii = np.minimum.reduce([centers[:,0], centers[:,1], 1 - centers[:,0], 1 - centers[:,1]])
        radii = np.clip(base_radii, 0, None)
        target_radii = 0.06 * size_factor
        radii = np.minimum(radii, target_radii)

        radii = compute_max_radii(centers, max_iters=10)

        alpha = 0.03
        momentum = 0.65
        prev_update = np.zeros_like(centers)
        max_iters = 800

        for it in range(max_iters):
            forces = compute_forces(centers, radii)
            centers, prev_update = update_centers(centers, forces, alpha, momentum, prev_update)
            prev_radii = radii.copy()
            radii = compute_max_radii(centers, max_iters=5)
            radii = np.minimum(radii * 1.002, target_radii)
            alpha *= 0.9985
            momentum *= 0.999
            if it > 400 and np.sum(np.abs(radii - prev_radii)) < 1e-6:
                break

        sum_radii = np.sum(radii)
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_centers = centers.copy()
            best_radii = radii.copy()

    return best_centers, best_radii
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
