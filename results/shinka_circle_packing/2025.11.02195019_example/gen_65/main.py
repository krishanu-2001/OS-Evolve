# EVOLVE-BLOCK-START
"""Modular stage-based scheduler for hybrid circle packing (n=26, unit square)"""

import numpy as np

def construct_packing():
    """
    Top-level: runs a modular, staged optimization scheduler to find a high-sum-of-radii packing of 26 circles in [0, 1]^2.
    Returns: (centers, radii)
    """
    n = 26
    margin = 0.02
    rng = np.random.default_rng(202406)
    # Candidate hexagon layouts to explore structural variety
    layouts = [
        [6,5,6,5,4],
        [5,6,5,6,4],
        [6,6,5,5,4],
        [5,5,6,6,4],
        [6,5,5,6,4]
    ]
    # Stage configuration
    num_candidates = 15
    hill_iters = 1600
    anneal_iters = 700
    physics_steps = 30
    polish_iters = 230

    # ====== Stage 1: Candidate set expansion ======
    candidate_centers = []
    candidate_sums = []
    for _ in range(num_candidates):
        if rng.random() < 0.6:
            row_counts = layouts[rng.integers(len(layouts))]
            centers = _hex_layout(row_counts, n, margin, rng)
        else:
            centers = rng.uniform(margin, 1-margin, size=(n,2))
        radii = compute_max_radii(centers, margin)
        candidate_centers.append(centers)
        candidate_sums.append(radii.sum())
    # Take the best initial candidate
    idx = np.argmax(candidate_sums)
    centers = candidate_centers[idx]

    # ====== Stage 2: Hill climbing (decayed step, multi-center perturb) ======
    centers = _hill_climb_stage(centers, hill_iters, margin, layouts, rng)

    # ====== Stage 3: Physics-inspired relaxation ======
    centers = _physics_stage(centers, physics_steps, margin, rng)

    # ====== Stage 4: Simulated Annealing ======
    centers = _anneal_stage(centers, anneal_iters, margin, layouts, rng)

    # ====== Stage 5: Final local polishing (mini-annealing, high temperature, small step) ======
    centers = _final_polish_stage(centers, polish_iters, margin, layouts, rng)

    # ====== Stage 6: Local greedy repacking sweep ======
    centers = _local_greedy_repack_stage(centers, margin, n_iter=1, rng=rng)

    radii = compute_max_radii(centers, margin)
    return centers, radii

# --- Stage helpers ---

def _hex_layout(row_counts, n, margin, rng):
    max_cols = max(row_counts)
    dx = (1 - 2*margin) / max_cols
    h  = dx * np.sqrt(3) / 2
    centers = np.zeros((n, 2))
    idx = 0
    for rid, cnt in enumerate(row_counts):
        x0 = margin + (max_cols - cnt) * dx / 2
        y = margin + rid*h
        for c in range(cnt):
            centers[idx] = [x0 + c*dx, y]
            idx += 1
    # Center the pattern vertically if <1 in height
    height = h * (len(row_counts) - 1)
    offset = (1 - 2*margin - height) * 0.5
    if offset > 0:
        centers[:,1] += offset
    # Tiny jitter to avoid degenerate initial overlaps
    centers += rng.normal(0, 0.0002, centers.shape)
    return centers

def _hill_climb_stage(centers, iters, margin, layouts, rng):
    n = centers.shape[0]
    dx = (1 - 2*margin) / max(map(len, layouts))
    alpha0 = dx * 0.55
    best_c = centers.copy()
    best_r = compute_max_radii(best_c, margin)
    best_s = best_r.sum()
    for t in range(iters):
        alpha = alpha0 * (1 - t / iters)
        cand_c = best_c.copy()
        prob = 0.47 * (1 - t/iters) + 0.11
        if rng.random() < prob:
            k = rng.integers(2, 6)
            idxs = rng.choice(n, size=k, replace=False)
            deltas = rng.uniform(-alpha, alpha, size=(k,2))
            cand_c[idxs] += deltas
        else:
            i = int(rng.integers(n))
            cand_c[i] += rng.uniform(-alpha, alpha, 2)
        np.clip(cand_c, margin, 1-margin, out=cand_c)
        cand_r = compute_max_radii(cand_c, margin)
        s = cand_r.sum()
        if s > best_s + 1e-12:
            best_s, best_c, best_r = s, cand_c, cand_r
    return best_c

def _physics_stage(centers, steps, margin, rng):
    n = centers.shape[0]
    c = centers.copy()
    r = compute_max_radii(c, margin)
    for _ in range(steps):
        forces = np.zeros_like(c)
        # Pairwise repulsion (if overlap)
        for i in range(n):
            for j in range(i+1, n):
                dv = c[j] - c[i]
                dist = np.linalg.norm(dv)
                min_d = r[i] + r[j] + 1e-6
                if dist < min_d:
                    dirv = dv/dist if dist > 1e-8 else rng.normal(size=2)
                    overlap = min_d - dist
                    f = 0.19 * overlap * dirv
                    forces[i] -= f
                    forces[j] += f
        # Boundary repulsion
        for i in range(n):
            x, y, ri = c[i,0], c[i,1], r[i]
            if x - ri < margin:
                forces[i, 0] += 0.22 * (margin - (x-ri))
            if x + ri > 1-margin:
                forces[i, 0] -= 0.22 * ((x+ri)-(1-margin))
            if y - ri < margin:
                forces[i, 1] += 0.22 * (margin - (y-ri))
            if y + ri > 1-margin:
                forces[i, 1] -= 0.22 * ((y+ri)-(1-margin))
        c += 0.14 * forces
        np.clip(c, margin, 1-margin, out=c)
        r = compute_max_radii(c, margin)
    return c

def _anneal_stage(centers, iters, margin, layouts, rng):
    n = centers.shape[0]
    c = centers.copy()
    r = compute_max_radii(c, margin)
    e = r.sum()
    dx = (1 - 2*margin) / max(map(len, layouts))
    T0, T1 = 1.1e-2, 2.2e-4
    for k in range(iters):
        T = T0 * ((1-k/iters) + (T1/T0)*(k/iters))
        i = int(rng.integers(n))
        cand = c.copy()
        step = dx * (0.69 + 0.36*np.sin(k/41.9))  # small periodic vib
        delta = rng.uniform(-step, step, 2) * (1 - k/iters)
        cand[i] += delta
        np.clip(cand, margin, 1-margin, out=cand)
        cand_r = compute_max_radii(cand, margin)
        cand_e = cand_r.sum()
        dE = cand_e - e
        if dE > 0 or rng.random() < np.exp(dE / (T+1e-16)):
            c, r, e = cand, cand_r, cand_e
    return c

def _final_polish_stage(centers, iters, margin, layouts, rng):
    # Mini-annealing: higher T0, decayed fast, single or paired center moves for late local optima escape
    n = centers.shape[0]
    c = centers.copy()
    r = compute_max_radii(c, margin)
    e = r.sum()
    dx = (1 - 2*margin) / max(map(len, layouts))
    T0, T1 = 5.5e-3, 1e-5
    for k in range(iters):
        T = T0 * ((1-k/iters) + (T1/T0)*(k/iters))
        cand = c.copy()
        if rng.random() < 0.16:
            # paired move
            idxs = rng.choice(n, size=2, replace=False)
            delta = rng.uniform(-dx, dx, (2,2)) * (1-k/iters)
            cand[idxs] += delta
        else:
            i = int(rng.integers(n))
            delta = rng.uniform(-dx, dx, 2) * (1-k/iters)
            cand[i] += delta
        np.clip(cand, margin, 1-margin, out=cand)
        cand_r = compute_max_radii(cand, margin)
        cand_e = cand_r.sum()
        dE = cand_e - e
        if dE > 0 or rng.random() < np.exp(dE / (T+1e-16)):
            c, r, e = cand, cand_r, cand_e
    return c

def _local_greedy_repack_stage(centers, margin, n_iter=1, rng=None):
    """
    For each circle, locally maximize its radius by grid+random sampling, fixing others.
    """
    c = centers.copy()
    n = c.shape[0]
    for _ in range(n_iter):
        for i in range(n):
            arr_c = np.delete(c, i, axis=0)
            arr_r = compute_max_radii(arr_c, margin)
            best_r = -1.0
            best_p = c[i].copy()
            # grid + random
            grid = np.linspace(margin, 1-margin, 5)
            pts = np.stack(np.meshgrid(grid,grid), -1).reshape(-1,2)
            if rng is not None:
                pts = np.vstack([pts, rng.uniform(margin,1-margin,(20,2))])
            else:
                pts = np.vstack([pts, np.random.uniform(margin,1-margin,(20,2))])
            for p in pts:
                # wall limit
                rr = min(p[0]-margin, p[1]-margin, 1-margin-p[0], 1-margin-p[1])
                if arr_c.size:
                    d = np.linalg.norm(arr_c - p,axis=1) - arr_r
                    rr = min(rr, d.min())
                if rr > best_r:
                    best_r, best_p = rr, p
            c[i] = best_p
    return c

def compute_max_radii(centers, margin=0.0):
    """
    Compute the maximal radii for given (n,2) centers, no-overlap and margin.
    Refined: Iterates to full convergence not just fixed passes.
    """
    n = centers.shape[0]
    radii = np.minimum.reduce([centers[:,0]-margin, centers[:,1]-margin,
                               1-margin-centers[:,0], 1-margin-centers[:,1]])
    radii = np.clip(radii, 0, 1)
    for _ in range(100):
        old = radii.copy()
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d <= 1e-10:
                    if radii[i]>0 or radii[j]>0:
                        radii[i]=radii[j]=0.0
                else:
                    ri, rj = radii[i], radii[j]
                    if ri + rj > d:
                        scale = d / (ri + rj)
                        radii[i] *= scale
                        radii[j] *= scale
        if np.max(np.abs(radii-old)) < 1e-7:
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