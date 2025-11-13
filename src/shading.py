"""
shading.py — Implements f(x, y), index, and brightness calculations
for the Fast Spheres algorithm using precomputed constants.

This corresponds to equations (v) through (xvi) in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.constants import precompute_constants

def compute_shading(consts, size=200):
    """
    Compute shading using the Fast Spheres parabolic approximation.
    """
    E_C, F_C, G_C, H_C = consts["E_C"], consts["F_C"], consts["G_C"], consts["H_C"]
    C_C, radius, max_index = consts["C_C"], consts["radius"], consts["max_index"]

    # Generate a pixel grid
    y, x = np.ogrid[-radius:radius, -radius:radius]
    mask = x**2 + y**2 <= radius**2  # inside circle

    # Apply the parabolic approximation illumination function
    f_xy = E_C * (x**2 + y**2) + F_C * x + G_C * y + H_C

    # Apply the lighting model (Lambertian + clamped to non-negative)
    index = np.maximum(f_xy, 0)

    # Normalise safely to 0–max_index
    denom = np.max(index[mask]) if np.any(index[mask] > 0) else 1
    index = np.clip(index / denom * max_index, 0, max_index)

    # Create a grayscale image buffer
    brightness = np.zeros_like(index)
    brightness[mask] = index[mask]
    return brightness.astype(np.uint8)


if __name__ == "__main__":
    consts = precompute_constants((0.3, 0.4, 1.0), radius=50, max_index=255)
    img = compute_shading(consts, size=200)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("Fast Spheres — Directional Lighting (Prototype)")
    plt.show()
