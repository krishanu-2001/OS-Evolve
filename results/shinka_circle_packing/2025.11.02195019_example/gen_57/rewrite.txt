# EVOLVE-BLOCK-START
"""Hybrid circle packing optimizer combining simulated annealing and force refinement for n=26"""

import numpy as np
from typing import List, Callable, Tuple

class HybridCirclePacker:
    def __init__(self,
                 n: int = 26,
                 init_methods: List[Callable[[], np.ndarray]] = None,
                 max_sweeps: int = 30,
                 tol: float = 1e-7,
                 refine_iters: int = 1500,
                 T0: float = 1e-2,
                 alpha: float = 0.995,
                 sigma_base: float = 0.015,
                 sigma_multi: float = 0.03,
                 stagnation_limit: int = 200,
                 force_iters: int = 80,
                 force_lr: float = 0.01):
        self.n = n
        self.max_sweeps = max_sweeps
        self.tol = tol
        self.refine_iters = refine_iters
        self.T0 = T0
        self.alpha = alpha
        self.sigma_base = sigma_base
        self.sigma_multi = sigma_multi
        self.stagnation_limit = stagnation_limit
        self.force_iters = force_iters
        self.force_lr = force_lr

        # Use a hybrid set of initial layouts including well-tuned radial and multiple hex patterns
        self.init_methods = init_methods or [
            self._radial_layout,
            self._hex_layout([6,5,6,5,4]),
            self._hex_layout([5,6,5,6,4]),
        ]

    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        best_score = -np.inf
        best_c, best_r = None, None
        for init in self.init_methods:
            centers = init()
            centers = np.clip(centers, 0.0, 1.0)
            radii = self._compute_radii(centers)
            c_opt, r_opt = self._refine(centers, radii)
            score = r_opt.sum()
            if score > best_score:
                best_score, best_c, best_r = score, c_opt.copy(), r_opt.copy()
        return best_c, best_r

    def _compute_radii(self, centers: np.ndarray) -> np.ndarray:
        """Vectorized iterative radii computation limiting radii by borders and pairwise overlaps"""
        radii = np.minimum.reduce([
            centers[:,0], centers[:,1],
            1 - centers[:,0], 1 - centers[:,1]
        ])
        for _ in range(self.max_sweeps):
            diff = centers[:, None, :] - centers[None, :, :]
            D = np.linalg.norm(diff, axis=2) + np.eye(self.n)
            sumr = radii[:, None] + radii[None, :]
            scale = np.minimum(1.0, D / sumr)
            min_scale = scale.min(axis=1)
            new_r = radii * min_scale
            if np.max(np.abs(new_r - radii)) < self.tol:
                break
            radii = new_r
        return radii

    def _refine(self, centers: np.ndarray, radii: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated annealing with adaptive perturbations plus force-based relaxation"""
        best_c = centers.copy()
        best_r = radii.copy()
        best_score = best_r.sum()
        curr_c, curr_r, curr_score = best_c.copy(), best_r.copy(), best_score
        T = self.T0
        stagnation = 0

        for _ in range(self.refine_iters):
            if stagnation > self.stagnation_limit:
                p_multi = 0.5
                sigma = self.sigma_multi
            else:
                p_multi = 0.2
                sigma = self.sigma_base

            if np.random.rand() < p_multi:
                idx = np.random.choice(self.n, size=3, replace=False)
            else:
                idx = [np.random.randint(self.n)]

            c_new = curr_c.copy()
            c_new[idx] += np.random.randn(len(idx), 2) * sigma
            c_new = np.clip(c_new, 0.0, 1.0)

            r_new = self._compute_radii(c_new)
            score_new = r_new.sum()
            dE = score_new - curr_score

            if dE > 0 or np.random.rand() < np.exp(dE / T):
                curr_c, curr_r, curr_score = c_new, r_new, score_new
                if curr_score > best_score:
                    best_c, best_r, best_score = curr_c.copy(), curr_r.copy(), curr_score
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1

            T *= self.alpha

        # Final force-based relaxation to resolve overlaps and improve packing
        c_force, r_force = self._force_refine(best_c)
        if r_force.sum() > best_r.sum():
            best_c, best_r = c_force, r_force

        return best_c, best_r

    def _radial_layout(self) -> np.ndarray:
        c = np.zeros((self.n, 2))
        c[0] = [0.5, 0.5]
        r1, r2 = 0.28, 0.65
        for i in range(8):
            θ = 2*np.pi*i/8 + np.pi/16
            c[i+1] = [0.5 + r1*np.cos(θ), 0.5 + r1*np.sin(θ)]
        for i in range(13):
            θ = 2*np.pi*i/13 + np.pi/13
            c[i+9] = [0.5 + r2*np.cos(θ), 0.5 + r2*np.sin(θ)]
        corners = [(0.1,0.1),(0.9,0.1),(0.1,0.9),(0.9,0.9)]
        for k, p in enumerate(corners, start=22):
            c[k] = p
        return c

    def _hex_layout(self, rows: List[int]) -> Callable[[], np.ndarray]:
        def layout() -> np.ndarray:
            margin = 0.1
            max_row = max(rows)
            dx = (1-2*margin)/(max_row-1) if max_row > 1 else 1-2*margin
            dy = dx*np.sqrt(3)/2
            y0 = (1 - dy*(len(rows)-1)) / 2
            pts = []
            for i, cnt in enumerate(rows):
                y = y0 + i*dy
                row_w = dx*(cnt-1) if cnt>1 else 0
                x0 = (1 - row_w)/2
                xs = x0 + np.arange(cnt)*dx if cnt>1 else [x0]
                for x in xs:
                    pts.append((x, y))
            arr = np.array(pts)
            if len(arr) > self.n:
                return arr[:self.n]
            if len(arr) < self.n:
                pad = np.random.rand(self.n-len(arr),2)*(1-2*margin)+margin
                return np.vstack([arr, pad])
            return arr
        return layout

    def _force_refine(self, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c = centers.copy()
        for _ in range(self.force_iters):
            r = self._compute_radii(c)
            forces = np.zeros_like(c)
            diff = c[:,None,:] - c[None,:,:]
            dist = np.linalg.norm(diff, axis=2) + np.eye(self.n)
            sumr = r[:,None] + r[None,:]
            overlap = sumr - dist
            mask = overlap > 0
            dirs = np.zeros_like(diff)
            nz = dist > 0
            dirs[nz] = diff[nz]/dist[nz][...,None]
            f = overlap[...,None] * dirs
            forces -= np.sum(np.where(mask[...,None], f, 0), axis=1)
            forces += np.sum(np.where(mask[...,None], f, 0), axis=0)
            # border repulsion
            left  = np.where(c[:,0] < r, (r - c[:,0]), 0)
            right = np.where(1-c[:,0] < r, (r - (1-c[:,0])), 0)
            down  = np.where(c[:,1] < r, (r - c[:,1]), 0)
            up    = np.where(1-c[:,1] < r, (r - (1-c[:,1])), 0)
            forces[:,0] += left - right
            forces[:,1] += down - up
            c += self.force_lr * forces
            c = np.clip(c, 0.0, 1.0)
        return c, self._compute_radii(c)

def construct_packing() -> Tuple[np.ndarray, np.ndarray]:
    """Construct a 26‐circle packing in a unit square."""
    packer = HybridCirclePacker()
    return packer.optimize()
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii