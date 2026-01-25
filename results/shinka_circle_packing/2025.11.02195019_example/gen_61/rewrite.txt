# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """
    Top-level entry: runs the combined pipeline and returns the best centers and radii.
    """
    pipeline = CombinedPipeline(n=26, margin=0.02, rng_seed=123)
    centers, radii = pipeline.run()
    return centers, radii

class CombinedPipeline:
    def __init__(self, n, margin, rng_seed):
        self.n = n
        self.margin = margin
        self.rng = np.random.default_rng(rng_seed)
        # hex layouts summing to 26
        self.layouts = [
            [6,5,6,5,4],
            [5,6,5,6,4],
            [6,6,5,5,4],
            [5,5,6,6,4],
            [6,5,5,6,4]
        ]
        # algorithmic parameters
        self.num_restarts = 15
        self.physics_steps = 50
        self.hill_iters = 1500
        self.anneal_iters = 1000
        self.repack_sweeps = 2

    def run(self):
        best_sum = -np.inf
        best_c = None
        best_r = None
        for _ in range(self.num_restarts):
            c = self._initialize()
            r = compute_max_radii(c)
            c = self._physics_relax(c, r)
            c = self._hill_climb(c)
            c = self._simulated_annealing(c)
            c = self._local_greedy_repack(c, n_iter=self.repack_sweeps)
            r = compute_max_radii(c)
            total = r.sum()
            if total > best_sum:
                best_sum, best_c, best_r = total, c.copy(), r.copy()
        return best_c, best_r

    def _initialize(self):
        """
        Hybrid seeding: 33% hex, 33% corner-edge, 34% uniform random
        """
        p = self.rng.random()
        if p < 0.33:
            layout = self.rng.choice(self.layouts)
            return self._hex_layout(layout)
        elif p < 0.66:
            return self._corner_layout()
        else:
            return self.rng.uniform(self.margin, 1 - self.margin, size=(self.n, 2))

    def _hex_layout(self, rows):
        maxc = max(rows)
        dx = (1 - 2*self.margin) / maxc
        dy = dx * np.sqrt(3) / 2
        centers = np.zeros((self.n,2))
        idx = 0
        for i, cnt in enumerate(rows):
            x0 = self.margin + (maxc - cnt)*dx/2
            y = self.margin + i * dy
            for j in range(cnt):
                centers[idx] = [x0 + j*dx, y]
                idx += 1
        return centers

    def _corner_layout(self):
        """
        Place 4 near corners, distribute rest along edges
        """
        m = self.margin + 0.01
        corners = [
            [m, m], [1-m, m],
            [m, 1-m], [1-m, 1-m]
        ]
        centers = np.zeros((self.n,2))
        for i,c in enumerate(corners):
            centers[i] = c
        idx = 4
        rem = self.n - 4
        per = rem // 4
        extra = rem % 4
        # bottom, right, top, left edges
        for e in range(4):
            count = per + (1 if e < extra else 0)
            for k in range(count):
                t = (k+1)/(count+1)
                if e==0:
                    centers[idx] = [m + t*(1-2*m), m]
                elif e==1:
                    centers[idx] = [1-m, m + t*(1-2*m)]
                elif e==2:
                    centers[idx] = [1-m - t*(1-2*m), 1-m]
                else:
                    centers[idx] = [m, 1-m - t*(1-2*m)]
                idx += 1
        return centers

    def _physics_relax(self, centers, radii):
        """
        Physics-based repulsive relaxation with decaying step size.
        """
        c = centers.copy()
        alpha = 0.05
        for _ in range(self.physics_steps):
            forces = np.zeros_like(c)
            # pairwise repulsion
            for i in range(self.n):
                for j in range(i+1, self.n):
                    dvec = c[j] - c[i]
                    dist = np.hypot(dvec[0], dvec[1]) + 1e-8
                    min_d = radii[i] + radii[j]
                    if dist < min_d:
                        overlap = (min_d - dist)/dist
                        f = dvec * overlap
                        forces[i] -= f
                        forces[j] += f
            # border forces
            for i in range(self.n):
                x,y = c[i]
                r = radii[i]
                if x - r < self.margin:
                    forces[i,0] += (self.margin - (x - r))
                if x + r > 1 - self.margin:
                    forces[i,0] -= ((x + r) - (1 - self.margin))
                if y - r < self.margin:
                    forces[i,1] += (self.margin - (y - r))
                if y + r > 1 - self.margin:
                    forces[i,1] -= ((y + r) - (1 - self.margin))
            c += alpha * forces
            np.clip(c, self.margin, 1-self.margin, out=c)
            radii = compute_max_radii(c)
            alpha *= 0.97
        return c

    def _hill_climb(self, centers):
        """
        Greedy local search with decaying step size.
        """
        best_c = centers.copy()
        best_r = compute_max_radii(best_c)
        best_s = best_r.sum()
        alpha0 = ((1 - 2*self.margin)/max(map(len,self.layouts))) * 0.4
        for t in range(self.hill_iters):
            alpha = alpha0 * (1 - t/self.hill_iters)
            cand = best_c.copy()
            if self.rng.random() < 0.2:
                k = self.rng.integers(2,5)
                idxs = self.rng.choice(self.n, k, replace=False)
                for i in idxs:
                    cand[i] += self.rng.uniform(-alpha, alpha, 2)
            else:
                i = int(self.rng.integers(self.n))
                cand[i] += self.rng.uniform(-alpha, alpha, 2)
            np.clip(cand, self.margin, 1-self.margin, out=cand)
            r = compute_max_radii(cand)
            s = r.sum()
            if s > best_s:
                best_s, best_c, best_r = s, cand, r
        return best_c

    def _simulated_annealing(self, centers):
        """
        Single-center annealing to escape local minima.
        """
        c = centers.copy()
        r = compute_max_radii(c)
        e = r.sum()
        T0, T1 = 5e-3, 5e-5
        for k in range(self.anneal_iters):
            T = T0*(1 - k/self.anneal_iters) + T1*(k/self.anneal_iters)
            i = int(self.rng.integers(self.n))
            cand = c.copy()
            step = (1 - 2*self.margin)/max(map(len,self.layouts))
            delta = self.rng.standard_normal(2) * step * (1 - k/self.anneal_iters)
            cand[i] += delta
            np.clip(cand, self.margin, 1-self.margin, out=cand)
            r2 = compute_max_radii(cand)
            e2 = r2.sum()
            dE = e2 - e
            if dE > 0 or self.rng.random() < np.exp(dE/max(T,1e-12)):
                c, r, e = cand, r2, e2
        return c

    def _local_greedy_repack(self, centers, n_iter=1):
        """
        For each circle, sample a small grid + random points to relocate for max radius.
        """
        c = centers.copy()
        n = self.n
        for _ in range(n_iter):
            for i in range(n):
                arr_c = np.delete(c, i, axis=0)
                arr_r = compute_max_radii(arr_c)
                best_r = -1.0
                best_p = c[i].copy()
                # grid + random
                grid = np.linspace(self.margin, 1-self.margin, 5)
                pts = np.stack(np.meshgrid(grid,grid), -1).reshape(-1,2)
                pts = np.vstack([pts, self.rng.uniform(self.margin,1-self.margin,(20,2))])
                for p in pts:
                    # wall limit
                    rr = min(p[0]-self.margin, p[1]-self.margin,
                             1-self.margin-p[0], 1-self.margin-p[1])
                    if arr_c.size:
                        d = np.linalg.norm(arr_c - p,axis=1) - arr_r
                        rr = min(rr, d.min())
                    if rr > best_r:
                        best_r, best_p = rr, p
                c[i] = best_p
        return c

def compute_max_radii(centers):
    """
    Compute max non-overlapping radii by iterative relaxation.
    """
    n = centers.shape[0]
    xs, ys = centers[:,0], centers[:,1]
    radii = np.minimum.reduce([xs, ys, 1-xs, 1-ys])
    for _ in range(20):
        changed = False
        for i in range(n):
            for j in range(i+1,n):
                dxy = centers[i] - centers[j]
                d = np.hypot(dxy[0], dxy[1])
                if d < 1e-8:
                    if radii[i]>0 or radii[j]>0:
                        radii[i]=radii[j]=0.0
                        changed = True
                else:
                    ri, rj = radii[i], radii[j]
                    if ri + rj > d:
                        scale = d/(ri+rj)
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