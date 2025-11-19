# EVOLVE-BLOCK-START
"""
Combined hybrid circle packing optimizer for n=26:
- Radial, ring and hexagonal initial layouts (fixed + dynamic)
- Simulated annealing refinement with adaptive perturbations
- Final force-based relaxation
"""
import numpy as np
from typing import Callable, List, Tuple

class CombinedCirclePacker:
    def __init__(
        self,
        n: int = 26,
        max_sweeps: int = 30,
        tol: float = 1e-7,
        refine_iters: int = 1500,
        T0: float = 1e-2,
        alpha: float = 0.995,
        sigma_base: float = 0.015,
        sigma_multi: float = 0.03,
        stagnation_limit: int = 200,
        force_iters: int = 80,
        force_lr: float = 0.01,
    ):
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

        # Build initial layout methods
        self.init_methods: List[Callable[[], np.ndarray]] = [
            self._radial_layout,
            self._ring_layout,
            self._hex_layout([6,5,6,5,4]),
            self._hex_layout([5,6,5,6,4]),
        ]
        # Add top 2 dynamic hex partitions
        combos = self._generate_hex_row_combinations(self.n)
        for rows in combos[:2]:
            self.init_methods.append(self._hex_layout(rows))

    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        best_score = -np.inf
        best_c = None
        best_r = None
        for init in self.init_methods:
            centers = np.clip(init(), 0.0, 1.0)
            radii = self._compute_radii(centers)
            c_opt, r_opt = self._refine(centers, radii)
            score = r_opt.sum()
            if score > best_score:
                best_score = score
                best_c, best_r = c_opt.copy(), r_opt.copy()
        return best_c, best_r

    def _compute_radii(self, centers: np.ndarray) -> np.ndarray:
        # start from distance to borders
        radii = np.minimum.reduce([
            centers[:,0], centers[:,1],
            1-centers[:,0], 1-centers[:,1]
        ]).copy()
        for _ in range(self.max_sweeps):
            diff = centers[:,None,:] - centers[None,:,:]
            D = np.linalg.norm(diff,axis=2) + np.eye(self.n)
            sumr = radii[:,None] + radii[None,:]
            # scale factor ≤1 for each pair
            scale = np.minimum(1.0, D / sumr)
            min_scale = scale.min(axis=1)
            new_r = radii * min_scale
            if np.max(np.abs(new_r - radii)) < self.tol:
                break
            radii = new_r
        return radii

    def _refine(self, centers: np.ndarray, radii: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        best_c, best_r = centers.copy(), radii.copy()
        best_score = best_r.sum()
        curr_c, curr_r, curr_score = best_c.copy(), best_r.copy(), best_score
        T = self.T0
        stagn = 0
        for _ in range(self.refine_iters):
            # adapt perturbation
            if stagn > self.stagnation_limit:
                sigma, p_multi = self.sigma_multi, 0.5
            else:
                sigma, p_multi = self.sigma_base, 0.2

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
            if dE > 0 or np.random.rand() < np.exp(dE/T):
                curr_c, curr_r, curr_score = c_new, r_new, score_new
                if curr_score > best_score:
                    best_c, best_r, best_score = curr_c.copy(), curr_r.copy(), curr_score
                    stagn = 0
                else:
                    stagn += 1
            else:
                stagn += 1
            T *= self.alpha

        # final force relaxation
        c_f, r_f = self._force_refine(best_c)
        if r_f.sum() > best_r.sum():
            best_c, best_r = c_f, r_f
        return best_c, best_r

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
        c[25] = [0.1,0.1]
        return c

    def _hex_layout(self, rows: List[int]) -> Callable[[], np.ndarray]:
        def layout() -> np.ndarray:
            margin = 0.1
            maxr = max(rows)
            dx = (1-2*margin)/(maxr-1) if maxr>1 else 1-2*margin
            dy = dx*np.sqrt(3)/2
            y0 = (1 - dy*(len(rows)-1))/2
            pts = []
            for i,cnt in enumerate(rows):
                y = y0 + i*dy
                w = dx*(cnt-1) if cnt>1 else 0
                x0 = (1-w)/2
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

    def _generate_hex_row_combinations(self, n: int) -> List[List[int]]:
        results = []
        def backtrack(sofar,left,rows_left):
            if rows_left==1:
                if left>=1:
                    cand = sofar+[left]
                    if max(cand)-min(cand)<=2 and max(cand)>=4:
                        results.append(cand)
                return
            for v in range(1,left-rows_left+2):
                backtrack(sofar+[v],left-v,rows_left-1)
        for R in range(4,7):
            backtrack([],n,R)
        # unique & sorted by (range, -max)
        uniq = {tuple(sorted(r)): r for r in results}
        combos = list(uniq.values())
        combos.sort(key=lambda r:(max(r)-min(r), -max(r)))
        return combos

    def _force_refine(self, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c = centers.copy()
        for _ in range(self.force_iters):
            r = self._compute_radii(c)
            diff = c[:,None,:] - c[None,:,:]
            dist = np.linalg.norm(diff,axis=2) + np.eye(self.n)
            sumr = r[:,None] + r[None,:]
            overlap = sumr - dist
            mask = overlap>0
            dirs = np.zeros_like(diff)
            nz = dist>0
            dirs[nz] = diff[nz]/dist[nz][...,None]
            f = overlap[...,None]*dirs
            forces = -np.sum(np.where(mask[...,None],f,0),axis=1) \
                     +np.sum(np.where(mask[...,None],f,0),axis=0)
            # border repulsion
            left  = np.maximum(r - c[:,0], 0)
            right = np.maximum(r - (1-c[:,0]), 0)
            down  = np.maximum(r - c[:,1], 0)
            up    = np.maximum(r - (1-c[:,1]), 0)
            forces[:,0] += left - right
            forces[:,1] += down - up
            c += self.force_lr * forces
            c = np.clip(c,0.0,1.0)
        return c, self._compute_radii(c)

def construct_packing() -> Tuple[np.ndarray, np.ndarray]:
    """Construct a 26‐circle packing in a unit square."""
    packer = CombinedCirclePacker()
    return packer.optimize()
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii