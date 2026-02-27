"""
shading.py — Implements f(x, y), index, and brightness calculations
for the Fast Spheres algorithm using precomputed constants.

This corresponds to equations (v) through (xvi) in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.constants import precompute_constants


def _circle_half_widths(radius):
    """
    Build per-row half-widths using midpoint/Bresenham circle stepping.

    Returns an array where widths[abs(y)] gives max |x| on that row.
    """
    widths = np.full(radius + 1, -1, dtype=np.int32)

    x = int(radius)
    y = 0
    d = 1 - x

    while x >= y:
        if y <= radius:
            widths[y] = max(widths[y], x)
        if x <= radius:
            widths[x] = max(widths[x], y)

        y += 1
        if d < 0:
            d += 2 * y + 1
        else:
            x -= 1
            d += 2 * (y - x) + 1

    # Fill any gaps conservatively by propagating the next valid width.
    for i in range(radius - 1, -1, -1):
        if widths[i] < 0:
            widths[i] = widths[i + 1]

    return widths


def _compute_shading_direct(consts, base_color, k_D, k_A, gamma):
    E_C, F_C, G_C, H_C = consts["E_C"], consts["F_C"], consts["G_C"], consts["H_C"]
    radius, max_index = consts["radius"], consts["max_index"]
    z0 = consts.get("z0", 0)

    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x**2 + y**2 <= radius**2

    f_xy = E_C * (x**2 + y**2) + F_C * x + G_C * y + H_C
    index = np.maximum(f_xy, 0)

    denom = np.max(index[mask]) if np.any(index[mask] > 0) else 1
    index = np.clip(index / denom * max_index, 0, max_index)

    bright = np.zeros_like(index, dtype=np.float32)
    bright[mask] = np.clip(k_D * (index[mask] / max_index) + k_A, 0, 1)
    bright[mask] = bright[mask] ** (1.0 / gamma)

    z_val = ((x**2 + y**2 + z0 * radius - radius**2) / radius).astype(np.float32)

    image = np.zeros((index.shape[0], index.shape[1], 3), dtype=np.uint8)
    for c in range(3):
        image[..., c] = (base_color[c] * bright).astype(np.uint8)

    return image, z_val, mask


def _compute_shading_diff(consts, base_color, k_D, k_A, gamma):
    """
    Bresenham-bounded, x-difference-stepped raster of f(x, y) and z(x, y).
    """
    E_C, F_C, G_C, H_C = consts["E_C"], consts["F_C"], consts["G_C"], consts["H_C"]
    K_C, L_C = consts["K_C"], consts["L_C"]
    radius, max_index = int(consts["radius"]), consts["max_index"]
    z0 = consts.get("z0", 0)

    size = 2 * radius + 1
    center = radius

    f_vals = np.zeros((size, size), dtype=np.float32)
    z_vals = np.full((size, size), np.inf, dtype=np.float32)
    mask = np.zeros((size, size), dtype=bool)

    widths = _circle_half_widths(radius)
    inv_r = 1.0 / float(radius)
    z_bias = z0 * radius - radius * radius

    max_positive_f = 0.0

    for y in range(-radius, radius + 1):
        analytic_x_max = int(np.sqrt(radius * radius - y * y))
        x_max = max(int(widths[abs(y)]), analytic_x_max)
        x_left = -x_max

        # Seed at row start, then advance only by first/second differences.
        f_val = E_C * (x_left * x_left + y * y) + F_C * x_left + G_C * y + H_C
        n_val = 2 * E_C * x_left + L_C

        z_num = x_left * x_left + y * y + z_bias
        z_step = 2 * x_left + 1

        row = y + center
        for x in range(x_left, x_max + 1):
            if x * x + y * y > radius * radius:
                f_val += n_val
                n_val += K_C
                z_num += z_step
                z_step += 2
                continue

            col = x + center
            mask[row, col] = True
            f_vals[row, col] = f_val
            z_vals[row, col] = z_num * inv_r

            if f_val > max_positive_f:
                max_positive_f = f_val

            f_val += n_val
            n_val += K_C

            z_num += z_step
            z_step += 2

    denom = max_positive_f if max_positive_f > 0 else 1.0
    index = np.clip(np.maximum(f_vals, 0.0) / denom * max_index, 0.0, max_index)

    bright = np.zeros_like(index, dtype=np.float32)
    bright[mask] = np.clip(k_D * (index[mask] / max_index) + k_A, 0, 1)
    bright[mask] = bright[mask] ** (1.0 / gamma)

    image = np.zeros((size, size, 3), dtype=np.uint8)
    for c in range(3):
        image[..., c] = (base_color[c] * bright).astype(np.uint8)

    return image, z_vals, mask


def _compute_shading_diff_view(consts, base_color, k_D, k_A, gamma):
    """
    Optimised path for light in view direction (F_C ~= 0, G_C ~= 0).

    Computes values in one octant and mirrors them to the other seven.
    """
    E_C, H_C = consts["E_C"], consts["H_C"]
    radius, max_index = int(consts["radius"]), consts["max_index"]
    z0 = consts.get("z0", 0)

    size = 2 * radius + 1
    center = radius

    f_vals = np.zeros((size, size), dtype=np.float32)
    z_vals = np.full((size, size), np.inf, dtype=np.float32)
    mask = np.zeros((size, size), dtype=bool)

    widths = _circle_half_widths(radius)
    inv_r = 1.0 / float(radius)
    z_bias = z0 * radius - radius * radius

    for y in range(0, radius + 1):
        analytic_x_max = int(np.sqrt(radius * radius - y * y))
        x_max = max(int(widths[y]), analytic_x_max)
        if x_max < y:
            continue

        x = y
        d2 = x * x + y * y
        step = 2 * x + 1

        while x <= x_max:
            if x * x + y * y > radius * radius:
                d2 += step
                step += 2
                x += 1
                continue

            f_val = E_C * d2 + H_C
            z_val = d2 * inv_r + z_bias * inv_r

            mirrored = (
                (x, y), (y, x), (-x, y), (-y, x),
                (x, -y), (y, -x), (-x, -y), (-y, -x),
            )
            for mx, my in mirrored:
                row = my + center
                col = mx + center
                mask[row, col] = True
                f_vals[row, col] = f_val
                z_vals[row, col] = z_val

            d2 += step
            step += 2
            x += 1

    max_positive_f = np.max(np.maximum(f_vals[mask], 0.0)) if np.any(mask) else 0.0
    denom = max_positive_f if max_positive_f > 0 else 1.0
    index = np.clip(np.maximum(f_vals, 0.0) / denom * max_index, 0.0, max_index)

    bright = np.zeros_like(index, dtype=np.float32)
    bright[mask] = np.clip(k_D * (index[mask] / max_index) + k_A, 0, 1)
    bright[mask] = bright[mask] ** (1.0 / gamma)

    image = np.zeros((size, size, 3), dtype=np.uint8)
    for c in range(3):
        image[..., c] = (base_color[c] * bright).astype(np.uint8)

    return image, z_vals, mask


def compute_shading(
    consts,
    size=200,
    base_color=(200, 200, 200),
    k_D=1.0,
    k_A=0.1,
    gamma=1.0,
    return_z=False,
    method="direct",
):
    """
    Compute shading using the Fast Spheres parabolic approximation.

    Parameters
    ----------
    method : {"direct", "diff", "diff_view"}
        "direct" uses dense vectorised evaluation of f(x, y),
        "diff" uses Bresenham row bounds plus x-direction differencing,
        "diff_view" computes one octant and mirrors it (view-direction case).
    """
    _ = size  # kept for API compatibility with previous versions

    if method == "diff_view":
        image, z_val, mask = _compute_shading_diff_view(consts, base_color, k_D, k_A, gamma)
    elif method == "diff":
        image, z_val, mask = _compute_shading_diff(consts, base_color, k_D, k_A, gamma)
    else:
        image, z_val, mask = _compute_shading_direct(consts, base_color, k_D, k_A, gamma)

    if return_z:
        return image, z_val.astype(np.float32), mask
    return image


if __name__ == "__main__":
    consts = precompute_constants((0.3, 0.4, 1.0), radius=50, max_index=255)
    img, z, mask = compute_shading(
        consts,
        size=200,
        base_color=(200, 150, 80),
        k_D=1.0,
        k_A=0.1,
        gamma=2.2,
        return_z=True,
        method="diff",
    )

    # Normalise only for visualisation.
    z_vis = np.zeros_like(z, dtype=np.float32)
    z_min = np.min(z[mask])
    z_max = np.max(z[mask])
    if z_max > z_min:
        z_vis[mask] = (z[mask] - z_min) / (z_max - z_min)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Shaded Sphere")

    plt.subplot(1, 2, 2)
    plt.imshow(z_vis, cmap="viridis")
    plt.axis("off")
    plt.title("Depth Map (Visualised)")

    plt.show()
