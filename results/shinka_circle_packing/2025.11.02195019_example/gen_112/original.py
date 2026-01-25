# EVOLVE-BLOCK-START
import numpy as np
import math

def greedy_initial(n, samples=10000, seed=0):
    """
    Sequentially place n circles.
    At each step, sample 'samples' random points and pick the one
    with the largest feasible radius given existing circles.
    Additionally, after initial sampling, perform extra gap-filling sampling on
    boundaries and around placed circle edges to better fill edges/corners.
    """
    rnd = np.random.RandomState(seed)
    centers = []
    radii = []
    for k in range(n):
        pts = rnd.rand(samples,2)
        # --- Gap-filling candidates ---
        # (1) Sample random points along boundary if k > 6
        n_bdry = 150 if k > 6 else 0
        if n_bdry > 0:
            sides = rnd.randint(4, size=n_bdry)
            bdry_pts = np.zeros((n_bdry, 2))
            vals = rnd.rand(n_bdry)
            for j, s in enumerate(sides):
                if s == 0:   # left edge: (0, y)
                    bdry_pts[j] = [0, vals[j]]
                elif s == 1: # right edge: (1, y)
                    bdry_pts[j] = [1, vals[j]]
                elif s == 2: # bottom: (x, 0)
                    bdry_pts[j] = [vals[j], 0]
                else:        # top: (x, 1)
                    bdry_pts[j] = [vals[j], 1]
            pts = np.vstack([pts, bdry_pts])
        # (2) Sample points near edges of previously placed circles if k > 1
        if k > 1:
            arr_centers = np.array(centers)
            arr_radii = np.array(radii)
            edge_pts = []
            n_edge = 15  # per circle
            for ci, ri in zip(arr_centers, arr_radii):
                angles = rnd.rand(n_edge)*2*np.pi
                # slightly outside actual circle to favor filling gaps
                rp = ri + 0.01 + 0.03*rnd.rand(n_edge)
                xs = ci[0] + rp*np.cos(angles)
                ys = ci[1] + rp*np.sin(angles)
                local = np.stack([xs, ys], axis=1)
                # keep inside square
                local = local[(local[:,0]>=0) & (local[:,0]<=1) & (local[:,1]>=0) & (local[:,1]<=1)]
                edge_pts.append(local)
            if len(edge_pts):
                pts = np.vstack([pts] + edge_pts)

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

def simulated_annealing(centers, radii, iters=15000, T0=0.05, Tend=1e-5, seed=1):
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

    multi_prob = 0.12  # increased multi-circle move probability
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

def construct_packing():
    """
    Build and optimize 26-circle packing via greedy initialization
    followed by simulated annealing refinement.
    """
    n = 26
    # Phase 1: greedy placement with enhanced gap filling
    centers, radii = greedy_initial(n, samples=10000, seed=42)
    # Phase 2: simulated annealing with increased multi-circle move probability
    centers, radii = simulated_annealing(centers, radii,
                                         iters=15000,
                                         T0=0.05,
                                         Tend=1e-5,
                                         seed=999)
    return centers, radii

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii