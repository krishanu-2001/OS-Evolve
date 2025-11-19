# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """
    Construct a 26-circle packing in a unit square with improved performance.
    Returns centers (26x2 array) and radii (length 26 array).
    """
    n = 26
    packer = CirclePacker(n=n)
    centers, radii = packer.optimize()
    return centers, radii


class CirclePacker:
    def __init__(self, n=26):
        self.n = n
        self.margin = 0.01  # small margin from edges to avoid numerical issues
        self.max_iter_radius = 100
        self.radius_tol = 1e-8
        self.force_iters = 60
        self.force_lr = 0.004
        self.anneal_iters = 16000
        self.T0 = 0.04
        self.Tend = 1e-5
        self.multi_move_prob = 0.12
        self.multi_move_count = 3
        self.perturb_scale = 0.018

    def optimize(self):
        # Generate diverse initial layouts and pick best
        candidates = []
        for layout_func in [self._radial_layout, self._hex_layout_1, self._hex_layout_2]:
            centers = layout_func()
            centers = np.clip(centers, self.margin, 1 - self.margin)
            radii = self._compute_radii(centers)
            candidates.append((centers, radii, radii.sum()))
        # Random uniform start
        centers = np.random.uniform(self.margin, 1-self.margin, (self.n, 2))
        radii = self._compute_radii(centers)
        candidates.append((centers, radii, radii.sum()))

        # Select best initial candidate
        centers, radii, best_score = max(candidates, key=lambda x: x[2])

        # Simulated annealing refinement with multi-circle moves
        centers, radii = self._simulated_annealing(centers, radii)

        # Final physics-based force relaxation
        centers, radii = self._force_relax(centers)

        return centers, radii

    def _compute_radii(self, centers):
        """
        Iteratively compute max radii for centers to avoid overlaps and boundary violations.
        """
        n = self.n
        radii = np.minimum.reduce([
            centers[:, 0] - self.margin,
            centers[:, 1] - self.margin,
            1 - self.margin - centers[:, 0],
            1 - self.margin - centers[:, 1]
        ])
        radii = np.clip(radii, 0, None)

        for _ in range(self.max_iter_radius):
            prev = radii.copy()
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < 1e-14:
                        # Coincident centers: set radii zero
                        radii[i] = 0.0
                        radii[j] = 0.0
                        continue
                    if radii[i] + radii[j] > d:
                        scale = d / (radii[i] + radii[j])
                        radii[i] *= scale
                        radii[j] *= scale
            if np.max(np.abs(radii - prev)) < self.radius_tol:
                break
        return radii

    def _simulated_annealing(self, centers, radii):
        n = self.n
        T = self.T0
        decay = (self.Tend / self.T0) ** (1.0 / self.anneal_iters)
        best_centers = centers.copy()
        best_radii = radii.copy()
        best_score = radii.sum()
        curr_centers = centers.copy()
        curr_radii = radii.copy()
        curr_score = best_score

        for it in range(self.anneal_iters):
            if np.random.rand() < self.multi_move_prob:
                # Multi-circle move
                idxs = np.random.choice(n, self.multi_move_count, replace=False)
            else:
                idxs = [np.random.randint(n)]

            cand_centers = curr_centers.copy()
            perturb = np.random.randn(len(idxs), 2) * self.perturb_scale
            cand_centers[idxs] += perturb
            cand_centers = np.clip(cand_centers, self.margin, 1 - self.margin)

            # Compute new radii for moved circles only
            cand_radii = curr_radii.copy()
            for i in idxs:
                cand_radii[i] = self._compute_radius_at(i, cand_centers, cand_radii)

            if np.any(cand_radii[idxs] < 1e-10):
                # Invalid move: reject
                continue

            cand_score = curr_score - curr_radii[idxs].sum() + cand_radii[idxs].sum()
            delta = cand_score - curr_score
            if delta >= 0 or np.random.rand() < np.exp(delta / T):
                curr_centers = cand_centers
                curr_radii = cand_radii
                curr_score = cand_score
                if curr_score > best_score:
                    best_centers = curr_centers.copy()
                    best_radii = curr_radii.copy()
                    best_score = curr_score
            T *= decay

        return best_centers, best_radii

    def _compute_radius_at(self, i, centers, radii):
        """
        Compute max radius for circle i given other circles and boundaries.
        """
        x, y = centers[i]
        r = min(x - self.margin, y - self.margin, 1 - self.margin - x, 1 - self.margin - y)
        if len(centers) > 1:
            others = np.delete(centers, i, axis=0)
            other_r = np.delete(radii, i)
            dists = np.linalg.norm(others - centers[i], axis=1) - other_r
            r = min(r, dists.min())
        return max(r, 0.0)

    def _force_relax(self, centers):
        """
        Physics-inspired relaxation with repulsive forces between overlapping circles
        and boundary repulsion to improve packing.
        """
        c = centers.copy()
        n = self.n

        for _ in range(self.force_iters):
            r = self._compute_radii(c)
            forces = np.zeros_like(c)

            # Pairwise repulsive forces if overlapping
            for i in range(n):
                for j in range(i+1, n):
                    dv = c[j] - c[i]
                    dist = np.linalg.norm(dv)
                    if dist < 1e-14:
                        continue
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        direction = dv / dist
                        f = 0.2 * overlap * direction
                        forces[i] -= f
                        forces[j] += f

            # Boundary repulsion
            for i in range(n):
                x, y = c[i]
                ri = r[i]
                if x - ri < self.margin:
                    forces[i, 0] += 0.2 * (self.margin - (x - ri))
                if x + ri > 1 - self.margin:
                    forces[i, 0] -= 0.2 * ((x + ri) - (1 - self.margin))
                if y - ri < self.margin:
                    forces[i, 1] += 0.2 * (self.margin - (y - ri))
                if y + ri > 1 - self.margin:
                    forces[i, 1] -= 0.2 * ((y + ri) - (1 - self.margin))

            c += self.force_lr * forces
            c = np.clip(c, self.margin, 1 - self.margin)

        r_final = self._compute_radii(c)
        return c, r_final

    def _radial_layout(self):
        """
        Radial layout: one center circle, rings of circles around it, plus corner circles.
        """
        c = np.zeros((self.n, 2))
        c[0] = [0.5, 0.5]
        # Inner ring: 8 circles radius ~0.27
        r1 = 0.27
        for i in range(8):
            angle = 2 * np.pi * i / 8 + 0.1
            c[i+1] = [0.5 + r1 * np.cos(angle), 0.5 + r1 * np.sin(angle)]
        # Outer ring: 13 circles radius ~0.63
        r2 = 0.63
        for i in range(13):
            angle = 2 * np.pi * i / 13 + 0.05
            c[i+9] = [0.5 + r2 * np.cos(angle), 0.5 + r2 * np.sin(angle)]
        # 4 corners
        corners = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
        c[22:26] = corners
        return c

    def _hex_layout_1(self):
        """
        Hexagonal layout variant 1 with rows [6,5,6,5,4] circles.
        """
        rows = [6, 5, 6, 5, 4]
        return self._hex_layout(rows)

    def _hex_layout_2(self):
        """
        Hexagonal layout variant 2 with rows [5,6,5,6,4] circles.
        """
        rows = [5, 6, 5, 6, 4]
        return self._hex_layout(rows)

    def _hex_layout(self, rows):
        margin = self.margin + 0.01  # slightly larger margin for layout
        max_cols = max(rows)
        dx = (1 - 2 * margin) / (max_cols - 1) if max_cols > 1 else 0.5
        dy = dx * np.sqrt(3) / 2
        total_height = dy * (len(rows) - 1)
        y0 = (1 - total_height) / 2
        centers = []
        for i, count in enumerate(rows):
            y = y0 + i * dy
            row_width = (count - 1) * dx if count > 1 else 0
            x0 = (1 - row_width) / 2
            for j in range(count):
                x = x0 + j * dx
                # offset every other row for hex packing
                if i % 2 == 1:
                    x += dx / 2
                centers.append([x, y])
        centers = np.array(centers)
        # Trim or pad to n circles
        if len(centers) > self.n:
            centers = centers[:self.n]
        elif len(centers) < self.n:
            pad = np.random.uniform(margin, 1 - margin, (self.n - len(centers), 2))
            centers = np.vstack([centers, pad])
        return centers

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
