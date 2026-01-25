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

def simulated_annealing(centers, radii, iters=15000, T0=0.05, Tend=1e-5, seed=999):
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

def force_refine(centers, iterations=60, lr=0.015):
    """
    Apply force-based relaxation to improve circle packing.
    Forces push overlapping circles apart and push circles away from walls.
    """
    c = centers.copy()
    n = c.shape[0]
    for _ in range(iterations):
        r = np.minimum.reduce([c[:,0], c[:,1], 1-c[:,0], 1-c[:,1]])
        # Compute pairwise distances and overlaps
        diff = c[:,None,:] - c[None,:,:]
        dist = np.linalg.norm(diff, axis=2) + np.eye(n)
        sumr = r[:,None] + r[None,:]
        overlap = sumr - dist
        mask = overlap > 0
        dirs = np.zeros_like(diff)
        nz = dist > 0
        dirs[nz] = diff[nz]/dist[nz][...,None]
        f = overlap[...,None] * dirs
        forces = np.zeros_like(c)
        forces -= np.sum(np.where(mask[...,None], f, 0), axis=1)
        forces += np.sum(np.where(mask[...,None], f, 0), axis=0)
        # border repulsion
        left  = np.where(c[:,0] < r, (r - c[:,0]), 0)
        right = np.where(1-c[:,0] < r, (r - (1-c[:,0])), 0)
        down  = np.where(c[:,1] < r, (r - c[:,1]), 0)
        up    = np.where(1-c[:,1] < r, (r - (1-c[:,1])), 0)
        forces[:,0] += left - right
        forces[:,1] += down - up
        c += lr * forces
        c = np.clip(c, 0.0, 1.0)
    # Recompute radii after relaxation
    radii = np.array([compute_radius_at(i, c, np.zeros(n)) for i in range(n)])
    return c, radii

def construct_packing():
    """
    Build and optimize 26-circle packing via greedy initialization
    followed by simulated annealing refinement and force relaxation.
    """
    n = 26
    centers, radii = greedy_initial(n, samples=8000, seed=42)
    centers, radii = simulated_annealing(centers, radii,
                                         iters=15000,
                                         T0=0.05,
                                         Tend=1e-5,
                                         seed=999)
    centers, radii = force_refine(centers, iterations=60, lr=0.015)
    return centers, radii

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii