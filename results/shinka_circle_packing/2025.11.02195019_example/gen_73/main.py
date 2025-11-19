# EVOLVE-BLOCK-START
"""Hybrid pipeline: ring+hex, hex, random, with force relaxation, annealing, and greedy repack for n=26"""

import numpy as np

def construct_packing():
    """
    Pipeline: try ring+hex, hex layouts, and random initializations,
    select best, then optimize with hill climbing, annealing, physics, and greedy repack.
    Returns:
        centers: np.array (26,2)
        radii:   np.array (26,)
    """
    n = 26
    margin = 0.02
    rng = np.random.default_rng(12345)
    layouts = [
        [6,5,6,5,4],
        [5,6,5,6,4],
        [6,6,5,5,4],
        [5,5,6,6,4],
        [6,5,5,6,4]
    ]
    num_candidates = 18

    candidate_centers = []
    candidate_sums = []
    for i in range(num_candidates):
        mode = rng.random()
        if i == 0 or mode < 0.33:
            # Ring+hex hybrid initialization (from parent code)
            centers = _ring_hex_hybrid_init(n, margin, rng)
        elif mode < 0.66:
            # Hex layout
            row_counts = layouts[rng.integers(len(layouts))]
            centers = _hex_layout(row_counts, n, margin, rng)
        else:
            # Random
            centers = rng.uniform(margin, 1-margin, size=(n,2))
        # Quick greedy repack to improve initial candidate
        centers = _greedy_repack_stage(centers, margin, rng, samples=60)
        radii = compute_max_radii(centers, margin)
        candidate_centers.append(centers)
        candidate_sums.append(radii.sum())
    idx = np.argmax(candidate_sums)
    centers = candidate_centers[idx]

    # Stage 2: Hill climbing (decayed step, multi-center perturb)
    centers = _hill_climb_stage(centers, 1800, margin, layouts, rng)

    # Stage 3: Physics-inspired relaxation
    centers = _physics_stage(centers, 35, margin, rng)

    # Stage 4: Simulated Annealing
    centers = _anneal_stage(centers, 1100, margin, layouts, rng)

    # Stage 5: Final local greedy repacking sweep
    centers = _greedy_repack_stage(centers, margin, rng, samples=180)

    radii = compute_max_radii(centers, margin)
    return centers, radii

def _ring_hex_hybrid_init(n, margin, rng):
    """Hybrid: central, ring, corners, edges, then fill with hex grid."""
    centers = np.zeros((n,2))
    # 1. Place a central circle
    centers[0] = [0.5, 0.5]
    # 2. Place 8 in a main ring (radius chosen for good internal clearance, randomized)
    main_ring_r = 0.29 + rng.uniform(-0.02, 0.02)
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i+1] = [0.5 + main_ring_r * np.cos(angle), 0.5 + main_ring_r * np.sin(angle)]
    # 3. Place 4 in corners, ensure inside
    corners = np.array([[margin+0.03,margin+0.03], [margin+0.03,1-margin-0.03], [1-margin-0.03,margin+0.03], [1-margin-0.03,1-margin-0.03]])
    centers[9:13] = corners
    # 4. The next 8: edge-centered (well-inside border)
    edge_offset = margin + 0.07
    edges = np.array([
        [edge_offset,0.5], [0.5,edge_offset], [1-edge_offset,0.5], [0.5,1-edge_offset],
        [edge_offset,edge_offset],
        [1-edge_offset,edge_offset],
        [edge_offset,1-edge_offset],
        [1-edge_offset,1-edge_offset]
    ])
    centers[13:21] = edges
    # 5. Fill remaining positions (5 left) using a tightest-packed hex grid determination
    used = 21
    needed = n - used
    grid_m = 8
    dx = (1-2*margin-0.02)/(grid_m-1)
    dy = dx * np.sqrt(3)/2
    hex_candidates = []
    for i in range(grid_m):
        for j in range(grid_m):
            x = margin+0.01 + j*dx + (i%2)*(dx/2)
            y = margin+0.01 + i*dy
            if margin < x < 1-margin and margin < y < 1-margin:
                hex_candidates.append([x,y])
    hex_candidates = np.array(hex_candidates)
    # Exclude close points to placed circles
    dists = np.linalg.norm(hex_candidates[:,None,:] - centers[:used][None,:,:], axis=2).min(axis=1)
    idx = np.argsort(-dists)[:needed]
    centers[used:used+needed] = hex_candidates[idx]
    # Add slight random jitter to all points
    centers += (rng.random((n,2)) - 0.5)*0.018
    centers = np.clip(centers, margin+0.005, 1-margin-0.005)
    return centers

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

def _greedy_repack_stage(centers, margin, rng, samples=150):
    """
    Local greedy repacking sweep: for each circle, sample candidate positions to maximize its radius.
    """
    n = centers.shape[0]
    c = centers.copy()
    for i in range(n):
        best_r = -1.0
        best_p = c[i].copy()
        # fix all other circles
        others = np.delete(c, i, axis=0)
        other_r = compute_max_radii(others, margin)
        # generate candidates: half uniform global, half jittered local
        pts_global = rng.uniform(margin, 1-margin, size=(samples//2, 2))
        angles = rng.uniform(0, 2*np.pi, samples//2)
        offsets = rng.uniform(0, (1-2*margin)*0.1, samples//2)
        pts_local = c[i] + np.stack(( np.cos(angles)*offsets, np.sin(angles)*offsets ), axis=1)
        pts_local = np.clip(pts_local, margin, 1-margin)
        pts = np.vstack((pts_global, pts_local))
        for p in pts:
            # wall constraint
            r = min(p[0]-margin, p[1]-margin, 1-margin-p[0], 1-margin-p[1])
            # overlap constraint
            if others.shape[0] > 0:
                d = np.linalg.norm(others - p, axis=1) - other_r
                r = min(r, d.min())
            if r > best_r:
                best_r = r
                best_p = p
        c[i] = best_p
    return c

def compute_max_radii(centers, margin=0.0):
    """
    Compute the maximal radii for given (n,2) centers, no-overlap and margin.
    Robust: Iterates to full convergence (up to 70 passes).
    """
    n = centers.shape[0]
    radii = np.minimum.reduce([centers[:,0]-margin, centers[:,1]-margin,
                               1-margin-centers[:,0], 1-margin-centers[:,1]])
    radii = np.clip(radii, 0, 1)
    for _ in range(70):
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