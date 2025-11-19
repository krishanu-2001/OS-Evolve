# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""

import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # This revised strategy uses a hybrid grid-like and perimeter-filling approach
    # for n=26 circles, aiming for a denser and more evenly distributed packing.

    # 1. Main Grid (4x4 = 16 circles)
    # These form the central, denser part of the packing.
    # Spacing for a 4x4 grid, centered. Each coordinate is 1/8, 3/8, 5/8, 7/8.
    grid_start_offset = 1.0 / 8.0 # Starting position for the first circle center
    x_coords = np.linspace(grid_start_offset, 1.0 - grid_start_offset, 4)
    y_coords = np.linspace(grid_start_offset, 1.0 - grid_start_offset, 4)

    k = 0
    for x in x_coords:
        for y in y_coords:
            centers[k] = [x, y]
            k += 1
    # Circles 0-15 are now set.

    # 2. Edge Circles (8 circles)
    # Place circles along the mid-points of each quarter edge section.
    # These are positioned relatively close to the square's borders to maximize boundary filling.
    edge_offset = 0.07 # Distance from the square border for these circles
    # Along the bottom edge (y = edge_offset)
    centers[16] = [0.25, edge_offset]
    centers[17] = [0.75, edge_offset]
    # Along the top edge (y = 1 - edge_offset)
    centers[18] = [0.25, 1 - edge_offset]
    centers[19] = [0.75, 1 - edge_offset]
    # Along the left edge (x = edge_offset)
    centers[20] = [edge_offset, 0.25]
    centers[21] = [edge_offset, 0.75]
    # Along the right edge (x = 1 - edge_offset)
    centers[22] = [1 - edge_offset, 0.25]
    centers[23] = [1 - edge_offset, 0.75]
    # Circles 16-23 are now set.

    # Total circles so far: 16 (grid) + 8 (edge) = 24 circles.
    # Remaining: 26 - 24 = 2 circles.

    # 3. Corner Circles (2 circles)
    # Place the last two circles in opposite corners to strategically fill these regions.
    # Placing them in opposite corners tends to distribute them better initially
    # and reduces immediate high-density overlap with other fixed centers.
    corner_offset = 0.07 # Using same offset as edge_offset for consistency
    centers[24] = [corner_offset, corner_offset] # Bottom-left corner
    centers[25] = [1 - corner_offset, 1 - corner_offset] # Top-right corner
    # Circles 24-25 are now set. All 26 circles are explicitly placed.

    # Clip to ensure initial circle centers are strictly within the unit square [0.01, 0.99].
    centers = np.clip(centers, 0.01, 0.99)

    # Objective function for the optimizer: minimize -sum(radii)
    # The optimizer works on a flattened array of center coordinates.
    def objective_function(flat_centers):
        current_centers = flat_centers.reshape(n, 2)
        # Ensure centers stay within bounds during optimization iterations
        current_centers = np.clip(current_centers, 0.001, 0.999) # Use slightly tighter clip for robustness during intermediate steps
        current_radii = compute_max_radii(current_centers)
        return -np.sum(current_radii)

    # Define bounds for each coordinate of the centers
    # Each (x,y) pair needs bounds [0.01, 0.99]
    bounds = [(0.01, 0.99) for _ in range(n * 2)] # n circles, 2 coords each

    # Perform the optimization
    # L-BFGS-B is a good choice for bounded problems
    # jac=True or hess=True could be used for faster convergence if gradient/hessian is provided
    # For now, let's let scipy approximate it.
    optimized_result = minimize(
        objective_function,
        centers.flatten(), # Initial guess
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': False, 'maxiter': 500} # Increased maxiter for better convergence
    )

    if optimized_result.success:
        centers = optimized_result.x.reshape(n, 2)
        # Final clip after optimization to ensure strict bounds
        centers = np.clip(centers, 0.01, 0.99)
        radii = compute_max_radii(centers)
    else:
        # Fallback if optimization fails
        # print(f"Warning: Center optimization failed. Message: {optimized_result.message}")
        radii = compute_max_radii(centers) # Use initial centers if optimization fails

    return centers, radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    from scipy.optimize import linprog

    # Objective: Maximize sum(radii) => minimize -sum(radii)
    c = -np.ones(n)

    # Constraints: A_ub * r <= b_ub
    A_ub = []
    b_ub = []

    # 1. Distance to square borders
    for i in range(n):
        x, y = centers[i]
        # r_i <= x
        row_x = np.zeros(n)
        row_x[i] = 1
        A_ub.append(row_x)
        b_ub.append(x)

        # r_i <= 1 - x
        row_1_x = np.zeros(n)
        row_1_x[i] = 1
        A_ub.append(row_1_x)
        b_ub.append(1 - x)

        # r_i <= y
        row_y = np.zeros(n)
        row_y[i] = 1
        A_ub.append(row_y)
        b_ub.append(y)

        # r_i <= 1 - y
        row_1_y = np.zeros(n)
        row_1_y[i] = 1
        A_ub.append(row_1_y)
        b_ub.append(1 - y)

    # 2. Non-overlap constraints: r_i + r_j <= dist_ij
    for i in range(n):
        for j in range(i + 1, n):
            dist_ij = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # Avoid numerical issues with very small distances or overlapping initial centers
            # If centers are too close, make constraint very tight or skip
            if dist_ij < 1e-6: # Centers are practically on top of each other
                # This makes it impossible for r_i and r_j to be positive, forcing them to 0
                row_overlap = np.zeros(n)
                row_overlap[i] = 1
                row_overlap[j] = 1
                A_ub.append(row_overlap)
                b_ub.append(0) # radii must sum to 0
                continue

            row_overlap = np.zeros(n)
            row_overlap[i] = 1
            row_overlap[j] = 1
            A_ub.append(row_overlap)
            b_ub.append(dist_ij)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds for radii: r_i >= 0
    bounds = [(0, None) for _ in range(n)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if res.success:
        radii = res.x
    else:
        # If optimization fails, return zero radii or fall back to a heuristic
        # For now, return an array of zeros and print a warning
        # In a robust system, one might log this failure or try a different method
        # print(f"Warning: linprog failed to find optimal radii. Message: {res.message}")
        radii = np.zeros(n)

    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii