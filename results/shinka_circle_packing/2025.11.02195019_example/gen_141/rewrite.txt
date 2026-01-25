# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.spatial import Voronoi

def compute_radius_at(i, centers, radii):
    """
    Given center i, compute its maximal radius not overlapping others or walls.
    """
    x, y = centers[i]
    r = min(x, y, 1-x, 1-y)
    if len(centers) > 1:
        others = np.delete(centers, i, axis=0)
        rads = np.delete(radii, i)
        d = np.linalg.norm(others - centers[i], axis=1) - rads
        r = min(r, d.min())
    return max(r, 0.0)

def voronoi_void_candidates(centers, radii, n_voids=20):
    """
    Given current circle centers, return up to n_voids Voronoi vertices
    that are inside the unit square and not inside any existing circle.
    """
    if len(centers) < 2:
        return np.empty((0,2))
    # Add mirrored points to avoid infinite Voronoi regions at the boundary
    mirrored = []
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            if dx==0 and dy==0: continue
            mirrored.append(centers + np.array([dx,dy]))
    pts = np.vstack([centers] + mirrored)
    try:
        vor = Voronoi(pts)
    except:
        return np.empty((0,2))
    # Only keep vertices inside [0,1]^2 and not inside any existing circle
    verts = vor.vertices
    mask = (verts[:,0]>=0) & (verts[:,0]<=1) & (verts[:,1]>=0) & (verts[:,1]<=1)
    verts = verts[mask]
    if len(verts)==0:
        return np.empty((0,2))
    arr_centers = np.array(centers)
    arr_radii = np.array(radii)
    # Remove points inside any existing circle (with a small epsilon)
    dists = np.linalg.norm(verts[None,:,:] - arr_centers[:,None,:], axis=2)
    min_dists = dists.min(axis=0)
    min_rads = arr_radii[np.argmin(dists, axis=0)]
    mask2 = min_dists > min_rads + 1e-5
    verts = verts[mask2]
    if len(verts)==0:
        return np.empty((0,2))
    # Sort by distance to nearest center (descending) to prioritize largest gaps
    dists_to_centers = np.linalg.norm(verts[None,:,:] - arr_centers[:,None,:], axis=2)
    min_dists2 = dists_to_centers.min(axis=0)
    idxs = np.argsort(-min_dists2)
    verts = verts[idxs]
    return verts[:n_voids]

def gap_voronoi_greedy_initial(n, base_samples=4000, local_samples=20, n_voids=20, seed=0):
    """
    Greedy initialization with:
    - Global random sampling
    - Local edge sampling
    - Voronoi void detection to target largest gaps
    """
    rnd = np.random.RandomState(seed)
    centers = []
    radii = []
    for k in range(n):
        pts = rnd.rand(base_samples,2)
        # Local edge sampling
        if k > 0:
            arr_centers = np.array(centers)
            arr_radii = np.array(radii)
            local_pts = []
            for c, r in zip(arr_centers, arr_radii):
                angles = rnd.rand(local_samples)*2*np.pi
                radii_local = r + 0.01 + rnd.rand(local_samples)*0.04
                xs = c[0] + radii_local * np.cos(angles)
                ys = c[1] + radii_local * np.sin(angles)
                pts_local = np.stack([xs, ys], axis=1)
                pts_local = pts_local[(pts_local[:,0]>=0) & (pts_local[:,0]<=1) & (pts_local[:,1]>=0) & (pts_local[:,1]<=1)]
                local_pts.append(pts_local)
            if local_pts:
                pts = np.vstack([pts] + local_pts)
        # Voronoi void candidates
        if k > 2:
            voids = voronoi_void_candidates(np.array(centers), np.array(radii), n_voids=n_voids)
            if len(voids)>0:
                pts = np.vstack([pts, voids])
        best_r = -1.0
        best_p = None
        if k == 0:
            xs = pts[:,0]; ys = pts[:,1]
            rs = np.minimum.reduce([xs, ys, 1-xs, 1-ys])
            idx = np.argmax(rs)
            best_r = rs[idx]; best_p = pts[idx]
        else:
            arr_centers = np.array(centers)
            arr_radii = np.array(radii)
            for p in pts:
                r = min(p[0], p[1], 1-p[0], 1-p[1])
                d = np.linalg.norm(arr_centers - p, axis=1) - arr_radii
                r = min(r, d.min())
                if r > best_r:
                    best_r = r; best_p = p
        centers.append(best_p)
        radii.append(max(best_r, 1e-8))
    return np.array(centers), np.array(radii)

def local_crowding(centers, radii, i, crowd_radius=0.18):
    """
    Estimate local crowding for circle i: number of other circles within crowd_radius.
    """
    dists = np.linalg.norm(centers - centers[i], axis=1)
    return np.sum((dists < crowd_radius) & (dists > 1e-8))

def adaptive_simulated_annealing(centers, radii, iters=18000, T0=0.07, Tend=1e-5, seed=1):
    """
    Simulated annealing with adaptive, per-circle step sizes based on local crowding.
    """
    rnd = np.random.RandomState(seed)
    n = centers.shape[0]
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = radii.sum()

    curr_centers = centers.copy()
    curr_radii = radii.copy()
    curr_sum = best_sum
    T = T0
    decay_base = (Tend / T0) ** (1.0 / iters)

    multi_prob = 0.08
    multi_count = 3
    base_step = 0.018
    min_step = 0.004
    max_step = 0.045

    stagnation = 0
    stagnation_limit = 200

    for it in range(iters):
        if stagnation < stagnation_limit:
            T = max(T * decay_base, Tend)
        else:
            T = max(T * (decay_base ** 3), Tend)

        if rnd.rand() < multi_prob:
            idxs = rnd.choice(n, multi_count, replace=False)
            old_ps = curr_centers[idxs].copy()
            old_rs = curr_radii[idxs].copy()
            # Step size per circle based on local crowding
            steps = []
            for idx in idxs:
                crowd = local_crowding(curr_centers, curr_radii, idx)
                step = base_step * (1.0/(1.0+crowd)) + min_step
                step = np.clip(step, min_step, max_step)
                steps.append(rnd.randn(2) * step)
            steps = np.array(steps)
            new_ps = old_ps + steps
            new_ps = np.clip(new_ps, 0.0, 1.0)
            curr_centers[idxs] = new_ps
            new_rs = np.array([compute_radius_at(i, curr_centers, curr_radii) for i in idxs])
            if (new_rs > 1e-8).all():
                new_sum = curr_sum - old_rs.sum() + new_rs.sum()
                delta = new_sum - curr_sum
                if delta >= 0 or rnd.rand() < math.exp(delta / T):
                    curr_radii[idxs] = new_rs
                    curr_sum = new_sum
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_centers[:] = curr_centers
                        best_radii[:] = curr_radii
                        stagnation = 0
                    else:
                        stagnation += 1
                else:
                    curr_centers[idxs] = old_ps
                    stagnation += 1
            else:
                curr_centers[idxs] = old_ps
                stagnation += 1
        else:
            i = rnd.randint(n)
            old_p = curr_centers[i].copy()
            old_r = curr_radii[i]
            crowd = local_crowding(curr_centers, curr_radii, i)
            step = base_step * (1.0/(1.0+crowd)) + min_step
            step = np.clip(step, min_step, max_step)
            move = rnd.randn(2) * step
            new_p = old_p + move
            new_p = np.clip(new_p, 0.0, 1.0)
            curr_centers[i] = new_p
            new_r = compute_radius_at(i, curr_centers, curr_radii)
            if new_r <= 1e-8:
                curr_centers[i] = old_p
                stagnation += 1
            else:
                new_sum = curr_sum - old_r + new_r
                delta = new_sum - curr_sum
                if delta >= 0 or rnd.rand() < math.exp(delta / T):
                    curr_radii[i] = new_r
                    curr_sum = new_sum
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_centers[:] = curr_centers
                        best_radii[:] = curr_radii
                        stagnation = 0
                    else:
                        stagnation += 1
                else:
                    curr_centers[i] = old_p
                    stagnation += 1

    return best_centers, best_radii

def force_based_micro_adjustment(centers, radii, n_steps=120, step_size=0.012):
    """
    Final micro-adjustment phase: apply repulsive/attractive forces to relieve local stresses.
    Only accept non-overlapping, in-boundary moves.
    """
    n = centers.shape[0]
    c = centers.copy()
    r = radii.copy()
    for _ in range(n_steps):
        for i in range(n):
            force = np.zeros(2)
            # Repulsion from other circles
            for j in range(n):
                if i==j: continue
                d = c[i] - c[j]
                dist = np.linalg.norm(d)
                min_dist = r[i] + r[j] + 1e-6
                if dist < min_dist and dist > 1e-8:
                    # Strong repulsion if overlapping
                    force += (d / dist) * (min_dist - dist) * 0.7
                elif dist < 0.22:
                    # Mild repulsion if close
                    force += (d / (dist+1e-8)) * 0.04
            # Attraction to center if near boundary
            for dim in range(2):
                if c[i][dim] < r[i] + 0.01:
                    force[dim] += (r[i] + 0.01 - c[i][dim]) * 0.3
                if c[i][dim] > 1 - r[i] - 0.01:
                    force[dim] -= (c[i][dim] - (1 - r[i] - 0.01)) * 0.3
            # Try move
            if np.linalg.norm(force) > 0:
                move = np.clip(force, -step_size, step_size)
                new_p = c[i] + move
                new_p = np.clip(new_p, 0.0, 1.0)
                old_p = c[i].copy()
                c[i] = new_p
                new_r = compute_radius_at(i, c, r)
                if new_r > 1e-8:
                    r[i] = new_r
                else:
                    c[i] = old_p
    # Final radii update
    for i in range(n):
        r[i] = compute_radius_at(i, c, r)
    return c, r

def local_greedy_repack(centers, radii, n_sweeps=2, local_samples=18):
    """
    After annealing, perform a local greedy repacking sweep:
    For each circle, sample candidates around current position and pick best feasible.
    """
    c = centers.copy()
    n = c.shape[0]
    for sweep in range(n_sweeps):
        for i in range(n):
            fixed = np.delete(c, i, axis=0)
            fixed_r = np.delete(radii, i)
            samples = [c[i]]
            for _ in range(local_samples):
                offset = 0.04 * (np.random.rand(2) - 0.5)
                candidate = c[i] + offset
                candidate = np.clip(candidate, 0.0, 1.0)
                samples.append(candidate)
            best_r, best_pos = -1, None
            for s in samples:
                r_max = min(s[0], s[1], 1-s[0], 1-s[1])
                dists = np.linalg.norm(fixed - s, axis=1) - fixed_r
                min_dist = dists.min() if dists.size > 0 else 1.0
                r_cand = min(r_max, min_dist)
                if r_cand > best_r:
                    best_r = r_cand
                    best_pos = s
            if best_r > 0:
                c[i] = best_pos
                radii[i] = best_r
    for i in range(n):
        radii[i] = compute_radius_at(i, c, radii)
    return c, radii

def construct_packing():
    """
    Build and optimize 26-circle packing via gap-driven Voronoi-guided greedy initialization,
    adaptive simulated annealing, local greedy repacking, and force-based micro-adjustment.
    """
    n = 26
    centers, radii = gap_voronoi_greedy_initial(n, base_samples=4000, local_samples=20, n_voids=22, seed=42)
    centers, radii = adaptive_simulated_annealing(centers, radii,
                                                  iters=18000,
                                                  T0=0.07,
                                                  Tend=1e-5,
                                                  seed=999)
    centers, radii = local_greedy_repack(centers, radii, n_sweeps=2, local_samples=18)
    centers, radii = force_based_micro_adjustment(centers, radii, n_steps=120, step_size=0.012)
    return centers, radii

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii