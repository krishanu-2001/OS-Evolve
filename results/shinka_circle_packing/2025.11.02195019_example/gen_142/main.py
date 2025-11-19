# EVOLVE-BLOCK-START
import numpy as np
import math

def construct_packing():
    n = 26
    np.random.seed(42)

    # 1. Structured initial placement
    centers = np.zeros((n,2))
    centers[0] = [0.5, 0.5]  # center circle

    main_ring_r = 0.29 + np.random.uniform(-0.02, 0.02)
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i+1] = [0.5 + main_ring_r * np.cos(angle), 0.5 + main_ring_r * np.sin(angle)]

    corners = np.array([[0.05,0.05], [0.05,0.95], [0.95,0.05], [0.95,0.95]])
    centers[9:13] = corners

    edge_offset = 0.09
    edges = np.array([
        [edge_offset,0.5], [0.5,edge_offset], [1-edge_offset,0.5], [0.5,1-edge_offset],
        [edge_offset,edge_offset], [1-edge_offset,edge_offset],
        [edge_offset,1-edge_offset], [1-edge_offset,1-edge_offset]
    ])
    centers[13:21] = edges

    # 2. Fill remaining with hex grid points
    used = 21
    needed = n - used
    grid_m = 8
    dx = 0.90 / (grid_m - 1)
    dy = dx * math.sqrt(3)/2
    candidates = []
    for i in range(grid_m):
        for j in range(grid_m):
            x = 0.05 + j*dx + (i%2)*(dx/2)
            y = 0.05 + i*dy
            if 0.04 < x < 0.96 and 0.04 < y < 0.96:
                candidates.append([x,y])
    candidates = np.array(candidates)
    # Exclude close points to existing circles
    dists = np.linalg.norm(candidates[:,None,:] - centers[:used][None,:,:], axis=2).min(axis=1)
    idxs = np.argsort(-dists)[:needed]
    centers[used:used+needed] = candidates[idxs]

    # 3. Add jitter for better fill
    centers += (np.random.rand(n,2)-0.5)*0.018
    centers = np.clip(centers, 0.015, 0.985)

    # 4. Initial radii via max radius computation
    radii = compute_max_radii(centers)

    # 5. Physics-inspired force relaxation to improve radii
    alpha = 0.035
    for it in range(400):
        forces = np.zeros_like(centers)
        # Overlap repulsion
        for i in range(n):
            for j in range(i+1,n):
                dxy = centers[i] - centers[j]
                dist = np.hypot(dxy[0], dxy[1]) + 1e-9
                allow = radii[i] + radii[j]
                if dist < allow and dist > 1e-10:
                    overlap = (allow - dist) / dist
                    f = dxy * overlap * 0.5
                    forces[i] += f
                    forces[j] -= f
        # Border correction
        margin = 0.001
        for i in range(n):
            x,y = centers[i]
            r = radii[i]
            if x - r < 0:
                forces[i,0] += (r - x + margin)
            if x + r > 1:
                forces[i,0] -= (x + r - 1 + margin)
            if y - r < 0:
                forces[i,1] += (r - y + margin)
            if y + r > 1:
                forces[i,1] -= (y + r - 1 + margin)
        # Update positions
        centers += alpha * forces
        centers = np.clip(centers, 0.01, 0.99)
        # Recompute radii
        radii = compute_max_radii(centers)
        # Decay step size
        alpha *= 0.992 if it < 250 else 0.998

    # 6. Adaptive simulated annealing with local perturbations
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    T0 = 0.0025
    temp = T0
    stagnation = 0
    n = centers.shape[0]
    for k in range(1500):
        # cluster-based move: pick a root and neighbors
        root = np.random.randint(n)
        dists = np.linalg.norm(centers - centers[root], axis=1)
        neighbors = np.argsort(dists)[1:4]
        idxs = np.concatenate(([root], neighbors))
        old_ps = centers[idxs].copy()
        old_rs = radii[idxs].copy()
        # adaptive step size based on circle radius
        max_r = radii.max() if radii.max() > 0 else 1.0
        adapt = (1 - radii[0]/max_r + 0.1)
        steps = np.random.randn(idxs.shape[0],2) * 0.008 * temp / (T0 + 1e-12) * adapt
        centers[idxs] = np.clip(old_ps + steps, 0.01, 0.99)
        new_rs = np.array([compute_radius_at(i, centers, radii) for i in idxs])
        if (new_rs > 1e-8).all():
            new_sum = np.sum(radii) - np.sum(old_rs) + np.sum(new_rs)
            delta = new_sum - np.sum(radii)
            if delta >= 0 or np.random.rand() < math.exp(delta / (temp + 1e-12)):
                radii[idxs] = new_rs
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_centers[:] = centers
                    best_radii[:] = radii
            else:
                centers[idxs] = old_ps
        else:
            centers[idxs] = old_ps

        # single circle move with adaptive step
        i = np.random.randint(n)
        old_p = centers[i].copy()
        old_r = radii[i]
        max_r = radii.max() if radii.max() > 0 else 1.0
        adapt = (1 - old_r / max_r + 0.1)
        step = np.random.randn(2) * 0.008 * temp / (T0 + 1e-12) * adapt
        new_p = np.clip(old_p + step, 0.01, 0.99)
        centers[i] = new_p
        new_r = compute_radius_at(i, centers, radii)
        if new_r <= 1e-8:
            centers[i] = old_p
        else:
            new_sum = np.sum(radii) - old_r + new_r
            delta = new_sum - np.sum(radii)
            if delta >= 0 or np.random.rand() < math.exp(delta / (temp + 1e-12)):
                radii[i] = new_r
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_centers[:] = centers
                    best_radii[:] = radii
            else:
                centers[i] = old_p
        # cooling schedule
        temp *= 0.997

    # 7. Local greedy repacking to refine radii
    def greedy_repack(c, samples=150):
        for i in range(c.shape[0]):
            radii_all = compute_max_radii(c)
            others = np.delete(c, i, axis=0)
            other_rs = np.delete(radii_all, i)
            best_p = c[i].copy()
            best_r = radii_all[i]
            # candidate points: jittered local + global
            local_pts = best_p + (np.random.randn(samples,2) * (best_r*0.4+1e-6))
            global_pts = np.random.rand(samples,2)*0.98+0.01
            pts = np.vstack((local_pts, global_pts))
            pts = np.clip(pts, 0.01, 0.99)
            for p in pts:
                r_new = min(p[0], 1-p[0], p[1], 1-p[1])
                if others.shape[0] > 0:
                    d = np.linalg.norm(others - p, axis=1) - other_rs
                    r_new = min(r_new, d.min())
                if r_new > best_r:
                    best_r = r_new
                    best_p = p
            c[i] = best_p
        return c
    centers = greedy_repack(centers)
    radii = compute_max_radii(centers)

    return centers, radii

def compute_max_radii(centers):
    n = centers.shape[0]
    xs, ys = centers[:,0], centers[:,1]
    radii = np.minimum.reduce([xs, ys, 1 - xs, 1 - ys])
    for _ in range(70):
        max_change = 0
        for i in range(n):
            for j in range(i+1, n):
                dxy = centers[i] - centers[j]
                dist = np.hypot(dxy[0], dxy[1])
                max_sum = radii[i] + radii[j]
                if max_sum > dist and dist > 1e-10:
                    scale = dist / max_sum
                    old_i, old_j = radii[i], radii[j]
                    radii[i] *= scale
                    radii[j] *= scale
                    max_change = max(max_change, abs(radii[i]-old_i), abs(radii[j]-old_j))
        if max_change < 1e-7:
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