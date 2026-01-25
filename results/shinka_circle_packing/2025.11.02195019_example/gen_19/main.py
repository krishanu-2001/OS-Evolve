# EVOLVE-BLOCK-START
"""Pipeline-based hybrid optimizer for n=26 circle packing"""

import numpy as np

def construct_packing():
    """
    Top-level entry: runs the packing pipeline and returns the best centers and radii.
    """
    pipeline = PackingPipeline(n=26, margin=0.02, rng_seed=123)
    centers, radii = pipeline.run()
    return centers, radii

class PackingPipeline:
    def __init__(self, n, margin, rng_seed):
        self.n = n
        self.margin = margin
        self.rng = np.random.default_rng(rng_seed)
        # Candidate hexagonal row layouts summing to n
        self.layouts = [
            [6,5,6,5,4],
            [5,6,5,6,4],
            [6,6,5,5,4],
            [5,5,6,6,4],
            [6,5,5,6,4]
        ]
        # Algorithmic parameters
        self.num_restarts = 20
        self.hill_iters = 2000
        self.anneal_iters = 1000
        self.physics_steps = 30

    def run(self):
        best_sum = -np.inf
        best_centers = None
        best_radii = None
        for _ in range(self.num_restarts):
            centers = self._initialize()
            centers = self._hill_climb(centers)
            centers = self._simulated_annealing(centers)
            centers = self._physics_relax(centers)
            radii = compute_max_radii(centers)
            s = radii.sum()
            if s > best_sum:
                best_sum, best_centers, best_radii = s, centers.copy(), radii.copy()
        return best_centers, best_radii

    def _initialize(self):
        # 50% hex layout, 50% random uniform
        if self.rng.random() < 0.5:
            layout = self.rng.choice(self.layouts)
            return self._hex_layout(layout)
        else:
            return self.rng.uniform(
                self.margin, 1 - self.margin, size=(self.n, 2)
            )

    def _hex_layout(self, row_counts):
        max_cols = max(row_counts)
        dx = (1 - 2*self.margin) / max_cols
        h = dx * np.sqrt(3) / 2
        centers = np.zeros((self.n, 2))
        idx = 0
        for rid, cnt in enumerate(row_counts):
            x0 = self.margin + (max_cols - cnt)*dx/2
            y = self.margin + rid * h
            for c in range(cnt):
                centers[idx] = [x0 + c*dx, y]
                idx += 1
        return centers

    def _hill_climb(self, centers):
        best_c = centers.copy()
        best_r = compute_max_radii(best_c)
        best_s = best_r.sum()
        alpha0 = ((1 - 2*self.margin) / max(map(len, self.layouts))) * 0.5
        for t in range(self.hill_iters):
            alpha = alpha0 * (1 - t / self.hill_iters)
            cand_c = best_c.copy()
            if self.rng.random() < 0.2:
                # multi-center move
                k = self.rng.integers(2, 5)
                idxs = self.rng.choice(self.n, size=k, replace=False)
                for i in idxs:
                    cand_c[i] += self.rng.uniform(-alpha, alpha, 2)
            else:
                i = int(self.rng.integers(self.n))
                cand_c[i] += self.rng.uniform(-alpha, alpha, 2)
            # clip into valid region
            np.clip(cand_c, self.margin, 1-self.margin, out=cand_c)
            cand_r = compute_max_radii(cand_c)
            s = cand_r.sum()
            if s > best_s:
                best_s, best_c, best_r = s, cand_c, cand_r
        return best_c

    def _simulated_annealing(self, centers):
        c = centers.copy()
        r = compute_max_radii(c)
        e = r.sum()
        T0, T1 = 1e-2, 1e-4
        for k in range(self.anneal_iters):
            T = T0 * ((1 - k/self.anneal_iters) + (T1/T0)*(k/self.anneal_iters))
            i = int(self.rng.integers(self.n))
            cand = c.copy()
            step = (1 - 2*self.margin) / max(map(len, self.layouts))
            delta = self.rng.uniform(-step, step, 2) * (1 - k/self.anneal_iters)
            cand[i] += delta
            np.clip(cand, self.margin, 1-self.margin, out=cand)
            cand_r = compute_max_radii(cand)
            cand_e = cand_r.sum()
            dE = cand_e - e
            if dE > 0 or self.rng.random() < np.exp(dE / T):
                c, r, e = cand, cand_r, cand_e
        return c

    def _physics_relax(self, centers):
        c = centers.copy()
        r = compute_max_radii(c)
        for _ in range(self.physics_steps):
            forces = np.zeros_like(c)
            # pairwise repulsion
            for i in range(self.n):
                for j in range(i+1, self.n):
                    dv = c[j] - c[i]
                    dist = np.linalg.norm(dv)
                    min_d = r[i] + r[j] + 1e-6
                    if dist < min_d:
                        dirv = dv/dist if dist>1e-8 else self.rng.normal(size=2)
                        overlap = min_d - dist
                        f = 0.2 * overlap * dirv
                        forces[i] -= f
                        forces[j] += f
            # boundary forces
            for i in range(self.n):
                x,y = c[i]; ri = r[i]
                if x - ri < self.margin: forces[i,0] += 0.2*(self.margin - (x-ri))
                if x + ri > 1-self.margin: forces[i,0] -= 0.2*((x+ri)-(1-self.margin))
                if y - ri < self.margin: forces[i,1] += 0.2*(self.margin - (y-ri))
                if y + ri > 1-self.margin: forces[i,1] -= 0.2*((y+ri)-(1-self.margin))
            c += 0.15 * forces
            np.clip(c, self.margin, 1-self.margin, out=c)
            r = compute_max_radii(c)
        return c

def compute_max_radii(centers):
    """
    Iterative relaxation enforcing border and pairwise constraints
    until convergence or max iterations.
    """
    n = centers.shape[0]
    # initial radii from borders
    radii = np.minimum.reduce([
        centers[:,0], centers[:,1],
        1-centers[:,0], 1-centers[:,1]
    ])
    for _ in range(20):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dv = centers[j] - centers[i]
                d = np.hypot(dv[0], dv[1])
                if d < 1e-8:
                    if radii[i]>0 or radii[j]>0:
                        radii[i]=radii[j]=0.0
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