# EVOLVE-BLOCK-START
import numpy as np
import math

def greedy_initial(n, samples=8000, seed=0):
    """
    Sequentially place n circles.
    At each step, sample 'samples' random points and pick the one
    with the largest feasible radius given existing circles.
    Additionally, for k>0, add a local sampling around existing circles
    to better fill gaps and edges.
    """
    rnd = np.random.RandomState(seed)
    centers = []
    radii = []
    for k in range(n):
        pts = rnd.rand(samples,2)
        # Add local samples near existing circles to fill gaps
        if k > 0:
            arr_centers = np.array(centers)
            arr_radii = np.array(radii)
            local_pts = []
            for c, r in zip(arr_centers, arr_radii):
                # sample points in annulus around circle edge
                angles = rnd.rand(20)*2*np.pi
                radii_local = r + rnd.rand(20)*0.05
                xs = c[0] + radii_local * np.cos(angles)
                ys = c[1] + radii_local * np.sin(angles)
                pts_local = np.stack([xs, ys], axis=1)
                # keep only points inside unit square
                pts_local = pts_local[(pts_local[:,0]>=0) & (pts_local[:,0]<=1) & (pts_local[:,1]>=0) & (pts_local[:,1]<=1)]
                local_pts.append(pts_local)
            if local_pts:
                pts = np.vstack([pts] + local_pts)
        best_r = -1.0
        best_p = None
        if k == 0:
            # for first circle just pick the best by walls
            xs = pts[:,0]; ys = pts[:,1]
            rs = np.minimum.reduce([xs, ys, 1-xs, 1-ys])
            idx = np.argmax(rs)
            best_r = rs[idx]; best_p = pts[idx]
        else:
            arr_centers = np.array(centers)
            arr_radii = np.array(radii)
            for p in pts:
                # radius limited by walls
                r = min(p[0], p[1], 1-p[0], 1-p[1])
                # limit by existing circles
                d = np.linalg.norm(arr_centers - p, axis=1) - arr_radii
                r = min(r, d.min())
                if r > best_r:
                    best_r = r; best_p = p
        centers.append(best_p)
        radii.append(max(best_r, 1e-6))
    return np.array(centers), np.array(radii)

def compute_radius_at(i, centers, radii):
    """
    Given center i, compute its maximal radius not overlapping others or walls.
    """
    x,y = centers[i]
    # wall limit
    r = min(x, y, 1-x, 1-y)
    if len(centers) > 1:
        # exclude self
        others = np.delete(centers, i, axis=0)
        rads  = np.delete(radii, i)
        d = np.linalg.norm(others - centers[i], axis=1) - rads
        r = min(r, d.min())
    return max(r, 0.0)

def simulated_annealing(centers, radii, iters=12000, T0=0.1, Tend=1e-4, seed=1):
    """
    Refine the placement by simulated annealing: perturb one or multiple circles at a time,
    recompute their max radii, and accept moves that improve total radius or
    stochastically based on temperature.
    """
    rnd = np.random.RandomState(seed)
    n = centers.shape[0]
    best_centers = centers.copy()
    best_radii  = radii.copy()
    best_sum = radii.sum()

    curr_centers = centers.copy()
    curr_radii = radii.copy()
    curr_sum = best_sum
    T = T0
    decay = (Tend/T0)**(1.0/iters)

    multi_prob = 0.05
    multi_count = 3
    step_scale = 0.02

    for it in range(iters):
        if rnd.rand() < multi_prob:
            # multi-circle perturbation
            idxs = rnd.choice(n, multi_count, replace=False)
            old_ps = curr_centers[idxs].copy()
            old_rs = curr_radii[idxs].copy()
            steps = rnd.randn(multi_count, 2) * step_scale
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
                else:
                    curr_centers[idxs] = old_ps
            else:
                curr_centers[idxs] = old_ps
        else:
            # single-circle perturbation
            i = rnd.randint(n)
            old_p = curr_centers[i].copy()
            old_r = curr_radii[i]
            step = rnd.randn(2) * step_scale
            new_p = old_p + step
            new_p = np.clip(new_p, 0.0, 1.0)
            curr_centers[i] = new_p
            new_r = compute_radius_at(i, curr_centers, curr_radii)
            if new_r <= 1e-8:
                curr_centers[i] = old_p
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
                else:
                    curr_centers[i] = old_p
        T *= decay

    return best_centers, best_radii

def local_greedy_repack(centers, radii, n_sweeps=3, local_samples=25):
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
            # Include current position as candidate
            samples = [c[i]]
            # Local random perturbations around current position
            for _ in range(local_samples):
                offset = 0.05 * (np.random.rand(2) - 0.5)
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
    # Recompute final radii for consistency
    for i in range(n):
        radii[i] = compute_radius_at(i, c, radii)
    return c, radii

def construct_packing():
    """
    Build and optimize 26-circle packing via greedy initialization,
    simulated annealing refinement, and final local greedy repacking.
    """
    n = 26
    # Phase 1: greedy placement
    centers, radii = greedy_initial(n, samples=8000, seed=42)
    # Phase 2: simulated annealing
    centers, radii = simulated_annealing(centers, radii,
                                         iters=15000,
                                         T0=0.05,
                                         Tend=1e-5,
                                         seed=999)
    # Phase 3: local greedy repacking
    centers, radii = local_greedy_repack(centers, radii, n_sweeps=3, local_samples=25)
    return centers, radii

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii