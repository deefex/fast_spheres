"""
diff_shading.py — Step B-1: 1D differencing along x for Fast Spheres.

This module compares:
- direct evaluation of f(x, y) = E_C (x^2 + y^2) + F_C x + G_C y + H_C
- vs. evaluation using x-direction difference stepping:

  f(x+1, y) = f(x, y) + N_C(x, y)
  N_C(x+1, y) = N_C(x, y) + K_C

where N_C(x, y) = 2 E_C x + L_C and K_C = 2 E_C, L_C = E_C + F_C.

We do this for every row y, but only use additions after the first x in each row.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.constants import precompute_constants


def compute_f_direct(consts):
    """
    Direct evaluation of f(x, y) over the full grid, for comparison.
    """
    E_C = consts["E_C"]
    F_C = consts["F_C"]
    G_C = consts["G_C"]
    H_C = consts["H_C"]
    radius = consts["radius"]

    # Coordinate grid centered at (0, 0)
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(x, y)

    f_direct = E_C * (X**2 + Y**2) + F_C * X + G_C * Y + H_C
    mask = X**2 + Y**2 <= radius**2
    return f_direct, mask, X, Y


def compute_f_diff_x(consts, X, Y):
    """
    Compute f(x, y) using only x-direction differencing, one row at a time.

    For each fixed y:
      - compute f at the leftmost x directly
      - compute N at that x: N = 2*E_C*x + L_C
      - sweep across the row using:
          f(x+1) = f(x) + N
          N      = N + K_C
    """
    E_C = consts["E_C"]
    F_C = consts["F_C"]
    G_C = consts["G_C"]
    H_C = consts["H_C"]
    K_C = consts["K_C"]
    L_C = consts["L_C"]

    radius = consts["radius"]
    x_vals = np.arange(-radius, radius + 1)
    y_vals = np.arange(-radius, radius + 1)

    f_diff = np.zeros_like(X, dtype=float)

    # Loop over rows (each fixed y)
    for j, y in enumerate(y_vals):
        x0 = x_vals[0]

        # f(x0, y) directly
        f_val = E_C * (x0**2 + y**2) + F_C * x0 + G_C * y + H_C

        # N_C(x0, y) = 2*E_C*x0 + L_C
        N = 2 * E_C * x0 + L_C

        f_diff[j, 0] = f_val

        # Sweep along +x using only adds
        for i in range(1, len(x_vals)):
            f_val += N
            N += K_C
            f_diff[j, i] = f_val

    return f_diff


def normalise_to_image(f, mask):
    """
    Map f values to [0, 255] grayscale within the mask.
    """
    # Only normalise where mask is True
    inside = f[mask]
    if inside.size == 0 or np.all(inside == 0):
        return np.zeros(f.shape, dtype=np.uint8)

    f_min = inside.min()
    f_max = inside.max()
    if f_max == f_min:
        return np.zeros(f.shape, dtype=np.uint8)

    norm = (f - f_min) / (f_max - f_min)
    norm[~mask] = 0.0
    img = (norm * 255).clip(0, 255).astype(np.uint8)
    return img

def compute_f_diff_y(consts, X, Y):
    """
    Compute f(x, y) using only y-direction differencing, one column at a time.

    For each fixed x:
      - compute f(x, y0) directly at the first y
      - compute S_C(x, y0) = 2*E_C*y0 + R_C
      - sweep along the column using:
          f(x, y+1) = f(x, y) + S
          S         = S + K_C
    """
    E_C = consts["E_C"]
    F_C = consts["F_C"]
    G_C = consts["G_C"]
    H_C = consts["H_C"]
    K_C = consts["K_C"]
    R_C = consts["R_C"]

    radius = consts["radius"]
    x_vals = np.arange(-radius, radius + 1)
    y_vals = np.arange(-radius, radius + 1)

    f_diff_y = np.zeros_like(X, dtype=float)

    # Loop over columns (each fixed x)
    for i, x in enumerate(x_vals):
        y0 = y_vals[0]

        # f(x, y0) directly
        f_val = E_C * (x**2 + y0**2) + F_C * x + G_C * y0 + H_C

        # S_C(x, y0) = 2*E_C*y0 + R_C
        S = 2 * E_C * y0 + R_C

        f_diff_y[0, i] = f_val

        # Sweep along +y using only adds
        for j in range(1, len(y_vals)):
            f_val += S
            S += K_C
            f_diff_y[j, i] = f_val

    return f_diff_y

def compute_f_diff_diag(consts, X, Y):
    """
    Compute f(x, y) using diagonal differencing only.

    Equation (xix):
        f(x+1, y+1) = f(x, y) + V
        V(x+1, y+1) = V + W_C

    With:
        V(x, y) = 2*E_C*(x + y) + (L_C + R_C)
    """
    E_C = consts["E_C"]
    F_C = consts["F_C"]
    G_C = consts["G_C"]
    H_C = consts["H_C"]
    L_C = consts["L_C"]
    R_C = consts["R_C"]
    W_C = consts["W_C"]

    radius = consts["radius"]

    x_vals = np.arange(-radius, radius + 1)
    y_vals = np.arange(-radius, radius + 1)

    f_diag = np.zeros_like(X, dtype=float)

    # Process diagonal lines starting at left border
    for y0_index, y0 in enumerate(y_vals):
        x = x_vals[0]
        y = y0

        # Compute f(x,y) directly for the starting point
        f_val = E_C*(x**2 + y**2) + F_C*x + G_C*y + H_C

        # Initial V_C(x,y) = 2*E_C*(x+y) + (L_C + R_C)
        V = 2*E_C*(x + y) + (L_C + R_C)

        f_diag[y0_index, 0] = f_val

        # Step along diagonal while inside grid
        xi, yi = 0, y0_index
        while xi+1 < len(x_vals) and yi+1 < len(y_vals):
            f_val += V
            V += W_C
            xi += 1
            yi += 1
            f_diag[yi, xi] = f_val

    # Now process diagonal lines starting at bottom border (excluding origin)
    for x0_index, x0 in enumerate(x_vals[1:], start=1):
        x = x0
        y = y_vals[0]

        f_val = E_C*(x**2 + y**2) + F_C*x + G_C*y + H_C
        V = 2*E_C*(x + y) + (L_C + R_C)

        f_diag[0, x0_index] = f_val

        xi, yi = x0_index, 0
        while xi+1 < len(x_vals) and yi+1 < len(y_vals):
            f_val += V
            V += W_C
            xi += 1
            yi += 1
            f_diag[yi, xi] = f_val

    return f_diag

def main():
    # Light from roughly viewer direction, slightly tilted
    consts = precompute_constants((0.3, 0.4, 1.0), radius=50, max_index=255)

    f_direct, mask, X, Y = compute_f_direct(consts)
    f_diff_x = compute_f_diff_x(consts, X, Y)
    f_diff_y = compute_f_diff_y(consts, X, Y)
    f_diff_diag = compute_f_diff_diag(consts, X, Y)

    # Compare numerically (inside the sphere only)
    diff_x = f_diff_x - f_direct
    diff_y = f_diff_y - f_direct
    diff_diag = f_diff_diag - f_direct

    max_abs_err_x = np.max(np.abs(diff_x[mask]))
    max_abs_err_y = np.max(np.abs(diff_y[mask]))
    max_abs_err_diag = np.max(np.abs(diff_diag[mask]))

    print(f"Max |f_diff_x - f_direct| inside sphere = {max_abs_err_x}")
    print(f"Max |f_diff_y - f_direct| inside sphere = {max_abs_err_y}")
    print(f"Max |f_diff_diag - f_direct| inside sphere = {max_abs_err_diag}")

    # Build grayscale images
    img_direct = normalise_to_image(f_direct, mask)

    img_diff_x = normalise_to_image(f_diff_x, mask)
    img_err_x = normalise_to_image(diff_x, mask)

    img_diff_y = normalise_to_image(f_diff_y, mask)
    img_err_y = normalise_to_image(diff_y, mask)

    img_diff_diag = normalise_to_image(f_diff_diag, mask)
    img_err_diag = normalise_to_image(diff_diag, mask)

    # Visual comparison
    plt.figure(figsize=(15, 15))

    # Row 1 — X differencing
    plt.subplot(3, 3, 1)
    plt.imshow(img_direct, cmap="gray", origin="lower")
    plt.title("Direct f(x, y)")
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.imshow(img_diff_x, cmap="gray", origin="lower")
    plt.title("Diff-based f(x, y) — X sweep")
    plt.axis("off")

    plt.subplot(3, 3, 3)
    plt.imshow(img_err_x, cmap="bwr", origin="lower")
    plt.title("Difference (X - direct)")
    plt.axis("off")

    # Row 2 — Y differencing
    plt.subplot(3, 3, 4)
    plt.imshow(img_direct, cmap="gray", origin="lower")
    plt.title("Direct f(x, y)")
    plt.axis("off")

    plt.subplot(3, 3, 5)
    plt.imshow(img_diff_y, cmap="gray", origin="lower")
    plt.title("Diff-based f(x, y) — Y sweep")
    plt.axis("off")

    plt.subplot(3, 3, 6)
    plt.imshow(img_err_y, cmap="bwr", origin="lower")
    plt.title("Difference (Y - direct)")
    plt.axis("off")

    # Row 3 — Diagonal differencing
    plt.subplot(3, 3, 7)
    plt.imshow(img_direct, cmap="gray", origin="lower")
    plt.title("Direct f(x, y)")
    plt.axis("off")

    plt.subplot(3, 3, 8)
    plt.imshow(img_diff_diag, cmap="gray", origin="lower")
    plt.title("Diff-based f(x, y) — Diagonal")
    plt.axis("off")

    plt.subplot(3, 3, 9)
    plt.imshow(img_err_diag, cmap="bwr", origin="lower")
    plt.title("Difference (Diag - direct)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
