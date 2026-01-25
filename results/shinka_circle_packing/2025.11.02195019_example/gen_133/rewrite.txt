# EVOLVE-BLOCK-START
import numpy as np
import math

def construct_packing():
    """
    Construct 26-circle packing by hybrid pipeline:
    diversified initialization, adaptive simulated annealing,
    and final physics relaxation with radius optimization.
    """
    pipeline = HybridPackingPipeline(n=26, margin=0.02, seed=1234)
    centers, radii = pipeline.run()
    return centers, radii

class HybridPackingPipeline:
    def __init__(self, n, margin, seed):
        self.n = n
        self.margin = margin
        self.rng = np.random.default_rng(seed)
        # Hex row layouts summing to n, balancing rows
        self.layouts = [
            [6,5,6,5,4],
            [5,6,5,6,4],
            [6,6,5,5,4],
            [5,5,6,6,4],
            [6,5,5,6,4]
        ]
        # Parameters
        self.num_restarts = 15
        self.hill_iters = 1500
        self.anneal_iters = 15000
        self.physics_steps = 40
        self.step_scale = 0.018
        self.multi_prob = 0.07
        self.multi_count = 3

    def run(self):
        best_sum = -np.inf
        best_centers = None
        best_radii = None
        for _ in range(self.num_restarts):
            c = self._initialize()
            c = self._hill_climb(c)
            c = self._adaptive_simulated_annealing(c)
            c = self._physics_relax(c)
            r = compute_max_radii(c)
            s = r.sum()
            if s > best_sum:
                best_sum, best_centers, best_radii = s, c.copy(), r.copy()
        return best_centers, best_radii

    def _initialize(self):
        p = self.rng.random()
        # 40% hex layout, 25% corner-edge layout, 35% random uniform
        if p < 0.4:
            layout = self.rng.choice(self.layouts)
            return self._hex_layout(layout)
        elif p < 0.65:
            return self._corner_edge_layout()
        else:
            return self.rng.uniform(self.margin, 1-self.margin, size=(self.n,2))

    def _hex_layout(self, rows):
        max_cols = max(rows)
        dx = (1 - 2*self.margin) / max_cols
        height = dx * np.sqrt(3) / 2
        centers = np.zeros((self.n, 2))
        idx = 0
        for r, count in enumerate(rows):
            x0 = self.margin + (max_cols - count)*dx/2
            y = self.margin + r*height
            for c in range(count):
                centers[idx] = [x0 + c*dx, y]
                idx += 1
        return centers

    def _corner_edge_layout(self):
        centers = np.zeros((self.n,2))
        m = self.margin + 0.01
        corners = np.array([
            [m,m],
            [1-m,m],
            [m,1-m],
            [1-m,1-m],
        ])
        centers[:4] = corners
        remaining = self.n - 4
        per_edge = remaining // 4
        extra = remaining % 4
        idx = 4
        # distribute along bottom edge, left to right
        for i in range(per_edge + (1 if extra>0 else 0)):
            t = (i+1)/(per_edge + (1 if extra>0 else 0) + 1)
            centers[idx] = [m + t*(1-2*m), m]
            idx +=1
        # right edge, bottom to top
        for i in range(per_edge + (1 if extra>1 else 0)):
            t = (i+1)/(per_edge + (1 if extra>1 else 0) + 1)
            centers[idx] = [1-m, m + t*(1-2*m)]
            idx +=1
        # top edge, right to left
        for i in range(per_edge + (1 if extra>2 else 0)):
            t = (i+1)/(per_edge + (1 if extra>2 else 0) + 1)
            centers[idx] = [1-m - t*(1-2*m), 1-m]
            idx +=1
        # left edge, top to bottom
        for i in range(per_edge):
            t = (i+1)/(per_edge + 1)
            if idx < self.n:
                centers[idx] = [m, 1-m - t*(1-2*m)]
                idx +=1
        return centers

    def _hill_climb(self, centers):
        c = centers.copy()
        r = compute_max_radii(c)
        best_sum = r.sum()
        best_c = c.copy()
        alpha0 = ((1 - 2*self.margin) / max(map(len, self.layouts))) * 0.5
        for t in range(self.hill_iters):
            alpha = alpha0 * (1-t/self.hill_iters)
            cand = best_c.copy()
            prob_multi = 0.5 * (1 - t/self.hill_iters) + 0.1
            if self.rng.random() < prob_multi:
                k = self.rng.integers(2,5)
                idxs = self.rng.choice(self.n, size=k, replace=False)
                for i in idxs:
                    cand[i] += self.rng.uniform(-alpha, alpha, 2)
            else:
                i = int(self.rng.integers(self.n))
                cand[i] += self.rng.uniform(-alpha, alpha, 2)
            np.clip(cand, self.margin, 1-self.margin, out=cand)
            cr = compute_max_radii(cand)
            s = cr.sum()
            if s > best_sum:
                best_sum = s
                best_c = cand
        return best_c

    def _adaptive_simulated_annealing(self, centers):
        c = centers.copy()
        r = compute_max_radii(c)
        e = r.sum()
        T0, Tend = 0.07, 1e-5
        iters = self.anneal_iters
        T = T0
        decay_base = (Tend / T0) ** (1.0 / iters)
        stagnation = 0
        stagnation_limit = 200
        for it in range(iters):
            # Adaptive cooling
            if stagnation < stagnation_limit:
                T = max(T * decay_base, Tend)
            else:
                T = max(T * (decay_base ** 3), Tend)

            if self.rng.random() < self.multi_prob:
                idxs = self.rng.choice(self.n, self.multi_count, replace=False)
                old_ps = c[idxs].copy()
                old_rs = compute_max_radii(c)[idxs].copy()
                steps = self.rng.normal(0, self.step_scale, (self.multi_count, 2))
                new_ps = old_ps + steps
                new_ps = np.clip(new_ps, self.margin, 1-self.margin)
                c[idxs] = new_ps
                new_rs = np.array([compute_radius_at(i, c, compute_max_radii(c)) for i in idxs])
                if (new_rs > 1e-8).all():
                    new_sum = compute_max_radii(c).sum()
                    delta = new_sum - e
                    if delta >= 0 or self.rng.random() < math.exp(delta / T):
                        r = compute_max_radii(c)
                        e = new_sum
                        stagnation = 0 if delta > 0 else stagnation+1
                    else:
                        c[idxs] = old_ps
                        stagnation += 1
                else:
                    c[idxs] = old_ps
                    stagnation += 1
            else:
                i = int(self.rng.integers(self.n))
                old_p = c[i].copy()
                old_r = compute_radius_at(i, c, compute_max_radii(c))
                step = self.rng.normal(0, self.step_scale, 2)
                new_p = old_p + step
                new_p = np.clip(new_p, self.margin, 1-self.margin)
                c[i] = new_p
                new_r = compute_radius_at(i, c, compute_max_radii(c))
                if new_r <= 1e-8:
                    c[i] = old_p
                    stagnation += 1
                else:
                    new_sum = compute_max_radii(c).sum()
                    delta = new_sum - e
                    if delta >= 0 or self.rng.random() < math.exp(delta / T):
                        e = new_sum
                        stagnation = 0 if delta > 0 else stagnation+1
                    else:
                        c[i] = old_p
                        stagnation += 1
        return c

    def _physics_relax(self, centers):
        c = centers.copy()
        r = compute_max_radii(c)
        for _ in range(self.physics_steps):
            forces = np.zeros_like(c)
            for i in range(self.n):
                for j in range(i+1, self.n):
                    dv = c[j] - c[i]
                    dist = np.linalg.norm(dv)
                    min_d = r[i] + r[j] + 1e-6
                    if dist < min_d:
                        dirv = dv/dist if dist>1e-8 else self.rng.normal(size=2)
                        overlap = min_d - dist
                        f = 0.25 * overlap * dirv
                        forces[i] -= f
                        forces[j] += f
            for i in range(self.n):
                x,y = c[i]
                ri = r[i]
                if x - ri < self.margin:
                    forces[i,0] += 0.25*(self.margin - (x - ri))
                if x + ri > 1 - self.margin:
                    forces[i,0] -= 0.25*((x + ri) - (1 - self.margin))
                if y - ri < self.margin:
                    forces[i,1] += 0.25*(self.margin - (y - ri))
                if y + ri > 1 - self.margin:
                    forces[i,1] -= 0.25*((y + ri) - (1 - self.margin))
            c += 0.12 * forces
            np.clip(c, self.margin, 1-self.margin, out=c)
            r = compute_max_radii(c)
        return c

def compute_radius_at(i, centers, radii):
    # Deterministic max radius for circle i given fixed others
    x, y = centers[i]
    r = min(x - 0, y - 0, 1 - x, 1 - y)
    if len(centers) > 1:
        others = np.delete(centers, i, axis=0)
        orads = np.delete(radii, i)
        dists = np.linalg.norm(others - centers[i], axis=1) - orads
        min_d = dists.min()
        r = min(r, min_d)
    return max(r, 0.0)

def compute_max_radii(centers):
    # Iterative radii relaxation for all circles simultaneously 
    n = centers.shape[0]
    radii = np.minimum.reduce([
        centers[:,0], centers[:,1],
        1-centers[:,0], 1-centers[:,1]
    ])
    for _ in range(25):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dv = centers[j] - centers[i]
                d = np.linalg.norm(dv)
                if d < 1e-8:
                    if radii[i] > 0 or radii[j] > 0:
                        radii[i] = 0.0
                        radii[j] = 0.0
                        changed = True
                else:
                    ri, rj = radii[i], radii[j]
                    if ri + rj > d:
                        scale = d / (ri + rj)
                        radii[i] *= scale
                        radii[j] *= scale
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