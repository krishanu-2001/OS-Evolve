# EVOLVE-BLOCK-START
"""Hybrid modular circle packing for n=26"""

import numpy as np

class PackingOptimizer:
    def __init__(self, n, max_sweeps=5, tol=1e-6):
        self.n = n
        self.max_sweeps = max_sweeps
        self.tol = tol

    def optimize(self):
        best_score = -np.inf
        best_centers = None
        best_radii = None
        # Try two different initial layouts
        for layout in (self._radial_layout, self._hex_layout):
            centers = layout()
            # Ensure inside the unit square
            centers = np.clip(centers, 0.0, 1.0)
            radii = self._iterative_refinement(centers)
            score = radii.sum()
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_radii = radii.copy()
        return best_centers, best_radii

    def _radial_layout(self):
        """Center + inner ring + outer ring + 4 corners = 26 circles"""
        n = self.n
        centers = np.zeros((n, 2))
        # Center circle
        centers[0] = [0.5, 0.5]
        # Inner ring of 8
        r1 = 0.28
        for i in range(8):
            theta = 2 * np.pi * i / 8 + np.pi / 16
            centers[i+1] = [0.5 + r1 * np.cos(theta), 0.5 + r1 * np.sin(theta)]
        # Outer ring of 13
        r2 = 0.65
        for i in range(13):
            theta = 2 * np.pi * i / 13 + np.pi / 13
            centers[i+9] = [0.5 + r2 * np.cos(theta), 0.5 + r2 * np.sin(theta)]
        # Four corner circles
        cm = 0.1
        corners = [(cm, cm), (1-cm, cm), (cm, 1-cm), (1-cm, 1-cm)]
        for idx, (x,y) in enumerate(corners, start=22):
            centers[idx] = [x, y]
        return centers

    def _hex_layout(self):
        """Variable-row hexagonal grid layout for 26 circles"""
        rows = [6, 5, 6, 5, 4]
        max_row = max(rows)
        margin = 0.1
        dx = (1 - 2*margin) / (max_row - 1)
        dy = dx * np.sqrt(3) / 2
        total_height = dy * (len(rows)-1)
        y0 = (1 - total_height) / 2
        centers = []
        for i, count in enumerate(rows):
            y = y0 + i * dy
            # center each row horizontally
            row_width = dx * (count - 1)
            x0 = (1 - row_width) / 2
            xs = x0 + np.arange(count) * dx
            for x in xs:
                centers.append((x, y))
        return np.array(centers)

    def _compute_border_radii(self, centers):
        """Initial radii limited by borders"""
        # dist to each border
        return np.min(np.vstack([centers.T, (1 - centers.T)]), axis=0)

    def _pairwise_scale(self, centers, radii):
        """One pass of pairwise scaling to resolve overlaps"""
        # Compute full distance matrix
        diff = centers[:, None, :] - centers[None, :, :]
        D = np.sqrt((diff**2).sum(axis=2)) + np.eye(self.n)  # add diag to avoid zero
        # Only consider i<j
        ri = radii[:, None]
        rj = radii[None, :]
        sumr = ri + rj
        overlap = sumr > D
        if not np.any(overlap):
            return radii
        # For each overlapping pair, compute scale factors
        scales = np.ones((self.n, self.n))
        scales[overlap] = D[overlap] / sumr[overlap]
        # Each circle's scale is the minimum over all its pairwise scales
        min_scales = scales.min(axis=1)
        return radii * min_scales

    def _iterative_refinement(self, centers):
        """Iteratively enforce border and pairwise constraints"""
        # Start with border-limited radii
        radii = self._compute_border_radii(centers)
        for _ in range(self.max_sweeps):
            new_radii = self._pairwise_scale(centers, radii)
            if np.max(np.abs(new_radii - radii)) < self.tol:
                break
            radii = new_radii
        return radii

def construct_packing():
    """
    Construct a 26‐circle packing in a unit square with maximized sum of radii.
    Returns centers (26×2 array) and radii (length 26 array).
    """
    optimizer = PackingOptimizer(n=26)
    return optimizer.optimize()

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii