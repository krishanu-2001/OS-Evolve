# EVOLVE-BLOCK-START
"""
Improved hybrid circle packing optimizer for n=26:
- Multiple fixed layouts for seeding
- Adaptive simulated annealing with multi-move perturbations
- Local greedy repack sweep
- Constraint-aware force-based fine-tuning
"""
import numpy as np
from typing import Tuple, List, Callable

class ImprovedCirclePacker:
    def __init__(self, n: int = 26):
        self.n = n
        # Annealing parameters
        self.max_iters = 4000
        self.T0 = 0.02
        self.alpha = 0.995
        self.stall_limit = 100
        self.sigma_base = 0.02
        self.sigma_multi = 0.04
        # Radii solver
        self.max_sweeps = 50
        self.tol = 1e-6
        # Local repack
        self.repack_passes = 2
        self.repack_samples = 30
        self.repack_radius = 0.12
        # Force refine
        self.force_iters = 60
        self.force_lr = 0.1

        # Initial layouts
        self.init_methods: List[Callable[[], np.ndarray]] = [
            self._hex_layout([6,5,6,5,4]),
            self._hex_layout([5,6,5,6,4]),
            self._radial_layout,
            self._ring_layout
        ]

    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        best_score = -np.inf
        best_c = None
        best_r = None
        # try each seed layout
        for init in self.init_methods:
            c0 = np.clip(init(), 0.0, 1.0)
            r0 = self._compute_radii(c0)
            c1, r1 = self._anneal_refine(c0, r0)
            c2, r2 = self._local_repack(c1)
            c3, r3 = self._force_refine(c2)
            score = np.sum(r3)
            if score > best_score:
                best_score, best_c, best_r = score, c3.copy(), r3.copy()
        return best_c, best_r

    def _compute_radii(self, centers: np.ndarray) -> np.ndarray:
        radii = np.minimum.reduce([
            centers[:,0], centers[:,1],
            1-centers[:,0], 1-centers[:,1]
        ]).copy()
        for _ in range(self.max_sweeps):
            diff = centers[:,None,:] - centers[None,:,:]
            D = np.linalg.norm(diff,axis=2) + np.eye(self.n)
            sumr = radii[:,None] + radii[None,:]
            scale = np.minimum(1.0, D / sumr)
            min_scale = scale.min(axis=1)
            new_r = radii * min_scale
            if np.max(np.abs(new_r - radii)) < self.tol:
                break
            radii = new_r
        return radii

    def _anneal_refine(self, centers: np.ndarray, radii: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
        curr_c, curr_r = centers.copy(), radii.copy()
        curr_score = curr_r.sum()
        best_c, best_r, best_score = curr_c.copy(), curr_r.copy(), curr_score
        T = self.T0
        stall = 0
        for it in range(self.max_iters):
            # adapt perturbation parameters
            if stall > self.stall_limit:
                p_multi, sigma = 0.5, self.sigma_multi
            else:
                p_multi, sigma = 0.2, self.sigma_base

            if np.random.rand() < p_multi:
                idx = np.random.choice(self.n, size=3, replace=False)
            else:
                idx = [np.random.randint(self.n)]
            c_new = curr_c.copy()
            c_new[idx] += np.random.randn(len(idx),2) * sigma
            c_new = np.clip(c_new, 0.0, 1.0)
            r_new = self._compute_radii(c_new)
            score_new = r_new.sum()
            dE = score_new - curr_score
            # acceptance
            if dE > 0 or np.random.rand() < np.exp(dE / T):
                curr_c, curr_r, curr_score = c_new, r_new, score_new
                if score_new > best_score:
                    best_c, best_r, best_score = c_new.copy(), r_new.copy(), score_new
                    stall = 0
                else:
                    stall += 1
            else:
                stall += 1
            # adaptive cooling: slow down if improving
            if stall < self.stall_limit:
                T *= self.alpha
            else:
                T *= self.alpha**1.5
        return best_c, best_r

    def _local_repack(self, centers: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
        c = centers.copy()
        for _ in range(self.repack_passes):
            for i in range(self.n):
                others = np.delete(c, i, axis=0)
                best_p = c[i].copy()
                best_r = 0.0
                # sample candidates around current
                probes = c[i] + (np.random.rand(self.repack_samples,2)*2-1)*self.repack_radius
                probes = np.vstack([probes, c[i]])
                probes = np.clip(probes, 0.0, 1.0)
                for p in probes:
                    border_r = min(p[0],p[1],1-p[0],1-p[1])
                    dmin = np.min(np.linalg.norm(others - p,axis=1))
                    cand_r = min(border_r, dmin)
                    if cand_r > best_r:
                        best_r, best_p = cand_r, p
                c[i] = best_p
        r = self._compute_radii(c)
        return c, r

    def _force_refine(self, centers: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
        c = centers.copy()
        for _ in range(self.force_iters):
            r = self._compute_radii(c)
            diff = c[:,None,:] - c[None,:,:]
            dist = np.linalg.norm(diff,axis=2) + np.eye(self.n)
            sumr = r[:,None] + r[None,:]
            overlap = sumr - dist
            mask = overlap > 0
            # compute unit directions
            dirs = np.zeros_like(diff)
            valid = dist > 0
            dirs[valid] = diff[valid] / dist[valid][...,None]
            # pairwise repulsion
            f = overlap[...,None] * dirs
            forces = -np.sum(np.where(mask[...,None], f, 0),axis=1) \
                     +np.sum(np.where(mask[...,None], f, 0),axis=0)
            # boundary repulsion
            left  = np.maximum(r - c[:,0], 0)
            right = np.maximum(r - (1-c[:,0]), 0)
            down  = np.maximum(r - c[:,1], 0)
            up    = np.maximum(r - (1-c[:,1]), 0)
            forces[:,0] += left - right
            forces[:,1] += down - up
            # update
            c += self.force_lr * forces
            c = np.clip(c, 0.0, 1.0)
        r_final = self._compute_radii(c)
        return c, r_final

    def _hex_layout(self, rows: List[int]) -> Callable[[], np.ndarray]:
        def layout() -> np.ndarray:
            margin = 0.05
            maxr = max(rows)
            dx = (1-2*margin)/(maxr-1) if maxr>1 else 1-2*margin
            dy = dx * np.sqrt(3)/2
            y0 = margin
            pts = []
            for i,cnt in enumerate(rows):
                y = y0 + i*dy
                w = dx*(cnt-1) if cnt>1 else 0
                x0 = margin + (maxr*dx - w)/2 - margin
                for j in range(cnt):
                    pts.append((x0 + j*dx, y))
            arr = np.array(pts)
            if len(arr)>self.n:
                return arr[:self.n]
            if len(arr)<self.n:
                pad = np.random.rand(self.n-len(arr),2)*(1-2*margin)+margin
                return np.vstack([arr,pad])
            return arr
        return layout

    def _radial_layout(self) -> np.ndarray:
        c = np.zeros((self.n,2))
        c[0] = [0.5,0.5]
        for i in range(8):
            θ = 2*np.pi*i/8 + np.pi/16
            c[i+1] = [0.5+0.28*np.cos(θ), 0.5+0.28*np.sin(θ)]
        for i in range(13):
            θ = 2*np.pi*i/13 + np.pi/13
            c[i+9] = [0.5+0.65*np.cos(θ), 0.5+0.65*np.sin(θ)]
        corners = [(0.1,0.1),(0.9,0.1),(0.1,0.9),(0.9,0.9)]
        for k,p in enumerate(corners, start=22):
            c[k] = p
        return c

    def _ring_layout(self) -> np.ndarray:
        c = np.zeros((self.n,2))
        c[0] = [0.5,0.5]
        for i in range(8):
            θ = 2*np.pi*i/8
            c[i+1] = [0.5+0.3*np.cos(θ), 0.5+0.3*np.sin(θ)]
        for i in range(13):
            θ = 2*np.pi*i/13
            c[i+9] = [0.5+0.65*np.cos(θ), 0.5+0.65*np.sin(θ)]
        # one corner to break symmetry
        c[25] = [0.1,0.1]
        return c

def construct_packing() -> Tuple[np.ndarray, np.ndarray]:
    """Construct a 26-circle packing in a unit square."""
    packer = ImprovedCirclePacker()
    return packer.optimize()
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii