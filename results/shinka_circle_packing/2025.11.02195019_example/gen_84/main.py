# EVOLVE-BLOCK-START
"""Modular multi-phase optimizer for n=26 circle packing with greedy gap-filling, simulated annealing, and force-based refinement."""

import numpy as np

class ModularCirclePacker:
    def __init__(self, n=26, margin=0.015, rng_seed=42):
        self.n = n
        self.margin = margin
        self.rng = np.random.default_rng(rng_seed)

    def construct(self):
        # 1. Try diverse initial layouts
        layouts = []
        layouts += self._generate_hex_layouts()
        layouts.append(self._generate_ring_layout())
        layouts.append(self._generate_corner_edge_layout())
        # 2. For each, perform greedy gap-filling sweep to locally maximize radii
        best_sum = -np.inf
        for centers in layouts:
            centers = self._greedy_gap_fill(centers)
            radii = self._compute_max_radii(centers)
            s = radii.sum()
            if s > best_sum:
                best_sum = s
                best_centers = centers.copy()
                best_radii = radii.copy()
        # 3. Simulated annealing with periodic cluster moves and adaptive schedule
        best_centers, best_radii = self._simulated_annealing(best_centers, best_radii)
        # 4. Final force-based repulsive relaxation
        best_centers = self._force_relax(best_centers, best_radii)
        best_radii = self._compute_max_radii(best_centers)
        return best_centers, best_radii

    def _generate_hex_layouts(self):
        # Try several plausible hex row arrangements for n=26
        candidate_row_layouts = [
            [6, 5, 6, 5, 4],
            [5, 6, 5, 6, 4],
            [6, 6, 5, 5, 4],
            [5, 5, 6, 6, 4],
            [6, 5, 5, 6, 4],
            [4, 6, 6, 6, 4],
            [4, 7, 5, 6, 4]
        ]
        layouts = []
        for row_counts in candidate_row_layouts:
            max_cols = max(row_counts)
            dx = (1 - 2*self.margin) / max_cols
            h  = dx * np.sqrt(3) / 2
            centers = np.zeros((self.n, 2))
            idx = 0
            for rid, cnt in enumerate(row_counts):
                x_start = self.margin + (max_cols - cnt) * dx / 2
                y = self.margin + rid * h
                for c in range(cnt):
                    if idx < self.n:
                        centers[idx, 0] = x_start + c * dx
                        centers[idx, 1] = y
                        idx += 1
            # If not enough, fill with random points
            if idx < self.n:
                centers[idx:] = self.rng.uniform(self.margin, 1-self.margin, size=(self.n-idx,2))
            layouts.append(centers)
        return layouts

    def _generate_ring_layout(self):
        # Place a central cluster, a ring, and a few on boundary
        n = self.n
        centers = np.zeros((n,2))
        centers[0] = [0.5, 0.5]
        for i in range(8):
            theta = 2*np.pi*i/8
            centers[i+1] = [0.5+0.26*np.cos(theta), 0.5+0.26*np.sin(theta)]
        for i in range(13):
            theta = 2*np.pi*i/13
            centers[i+9] = [0.5+0.62*np.cos(theta), 0.5+0.62*np.sin(theta)]
        # corners
        corners = [(self.margin, self.margin), (1-self.margin, self.margin), (self.margin, 1-self.margin), (1-self.margin, 1-self.margin)]
        for k, p in enumerate(corners, start=22):
            centers[k] = p
        return centers

    def _generate_corner_edge_layout(self):
        # Corners, edge-centers, and inner points
        n = self.n
        centers = np.zeros((n,2))
        # 4 corners
        centers[0] = [self.margin, self.margin]
        centers[1] = [1-self.margin, self.margin]
        centers[2] = [self.margin, 1-self.margin]
        centers[3] = [1-self.margin, 1-self.margin]
        # 4 edge-centers
        centers[4] = [0.5, self.margin]
        centers[5] = [self.margin, 0.5]
        centers[6] = [1-self.margin, 0.5]
        centers[7] = [0.5, 1-self.margin]
        # Rest: fill with a perturbed grid
        grid_pts = self._perturbed_grid(self.n-8)
        centers[8:] = grid_pts
        return centers

    def _perturbed_grid(self, m):
        # Uniform grid inside margin, with small random perturbation
        s = int(np.ceil(np.sqrt(m)))
        xs = np.linspace(self.margin, 1-self.margin, s)
        ys = np.linspace(self.margin, 1-self.margin, s)
        pts = np.array(np.meshgrid(xs, ys)).reshape(2,-1).T
        pts = pts[:m]
        pts += self.rng.uniform(-0.01, 0.01, pts.shape)
        pts = np.clip(pts, self.margin, 1-self.margin)
        return pts

    def _greedy_gap_fill(self, centers):
        # After initial placement, for each circle, try a local greedy search to improve its position/radius
        n = self.n
        for sweep in range(2):
            for i in range(n):
                best_c = centers[i].copy()
                best_r = self._compute_single_radius(i, centers)
                for _ in range(12):
                    cand = best_c + self.rng.uniform(-0.03,0.03,2)
                    cand = np.clip(cand, self.margin, 1-self.margin)
                    canders = centers.copy()
                    canders[i] = cand
                    r = self._compute_single_radius(i, canders)
                    if r > best_r:
                        best_c, best_r = cand, r
                centers[i] = best_c
        return centers

    def _compute_single_radius(self, idx, centers):
        # Compute the maximal radius for circle idx given all centers
        c = centers[idx]
        others = np.delete(centers, idx, axis=0)
        dists = np.linalg.norm(others-c, axis=1)
        min_dist = np.min(dists) if len(dists)>0 else 1.0
        min_border = min(c[0]-self.margin, c[1]-self.margin, 1-self.margin-c[0], 1-self.margin-c[1])
        r = min(min_border, min_dist/2)
        return max(r,0.0)

    def _compute_max_radii(self, centers):
        # Iterative update for all radii
        n = centers.shape[0]
        radii = np.minimum.reduce([
            centers[:,0]-self.margin,
            centers[:,1]-self.margin,
            1-self.margin-centers[:,0],
            1-self.margin-centers[:,1]
        ])
        for _ in range(15):
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(centers[i]-centers[j])
                    if d < 1e-10:
                        radii[i]=radii[j]=0.0
                        changed = True
                    else:
                        ri, rj = radii[i], radii[j]
                        if ri + rj > d:
                            scale = d / (ri + rj)
                            new_ri = ri * scale
                            new_rj = rj * scale
                            if new_ri < ri or new_rj < rj:
                                radii[i] = new_ri
                                radii[j] = new_rj
                                changed = True
            if not changed:
                break
        return radii

    def _simulated_annealing(self, centers, radii):
        n = self.n
        best_c = centers.copy()
        best_r = radii.copy()
        best_sum = best_r.sum()
        curr_c = best_c.copy()
        curr_r = best_r.copy()
        curr_sum = best_sum
        T = 0.02
        alpha = 0.995
        stagn = 0
        stagn_limit = 200
        max_iters = 4000
        for t in range(max_iters):
            # Adaptive temperature schedule
            if stagn > stagn_limit:
                T = min(T*1.05, 0.04)
            else:
                T *= alpha

            # 20%: cluster move, else single move
            if self.rng.uniform() < 0.20:
                count = self.rng.integers(2, 5)
                idx = self.rng.choice(n, size=count, replace=False)
            else:
                idx = [self.rng.integers(n)]
            cand_c = curr_c.copy()
            for i in idx:
                step = T * self.rng.normal(0, 0.5, 2)
                cand = cand_c[i] + step
                cand = np.clip(cand, self.margin, 1-self.margin)
                cand_c[i] = cand

            cand_r = self._compute_max_radii(cand_c)
            cand_sum = cand_r.sum()
            dE = cand_sum - curr_sum
            if dE > 0 or self.rng.uniform() < np.exp(dE/T):
                curr_c, curr_r, curr_sum = cand_c, cand_r, cand_sum
                if cand_sum > best_sum:
                    best_c, best_r, best_sum = cand_c.copy(), cand_r.copy(), cand_sum
                    stagn = 0
                else:
                    stagn += 1
            else:
                stagn += 1
        return best_c, best_r

    def _force_relax(self, centers, radii, steps=60):
        n = self.n
        c = centers.copy()
        for _ in range(steps):
            f = np.zeros_like(c)
            # Pairwise repulsion
            for i in range(n):
                for j in range(i+1,n):
                    d_vec = c[j]-c[i]
                    dist = np.linalg.norm(d_vec)
                    min_dist = radii[i]+radii[j]+1e-8
                    if dist < min_dist:
                        if dist > 1e-8:
                            direction = d_vec/dist
                        else:
                            direction = self.rng.uniform(-1,1,2)
                            direction /= np.linalg.norm(direction)
                        overlap = min_dist-dist
                        ff = 0.18*overlap*direction
                        f[i] -= ff
                        f[j] += ff
            # Border repulsion
            for i in range(n):
                x, y = c[i]
                r = radii[i]
                if x-r < self.margin:
                    f[i,0] += 0.2*(self.margin-(x-r))
                if x+r > 1-self.margin:
                    f[i,0] -= 0.2*((x+r)-(1-self.margin))
                if y-r < self.margin:
                    f[i,1] += 0.2*(self.margin-(y-r))
                if y+r > 1-self.margin:
                    f[i,1] -= 0.2*((y+r)-(1-self.margin))
            c += 0.13 * f
            c = np.clip(c, self.margin, 1-self.margin)
        return c

def construct_packing():
    packer = ModularCirclePacker()
    return packer.construct()

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii