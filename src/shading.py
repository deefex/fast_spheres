"""
shading.py — Implements f(x, y), index, and brightness calculations
for the Fast Spheres algorithm using precomputed constants.

This corresponds to equations (v) through (xvi) in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.constants import precompute_constants

def compute_shading(consts, size=200, base_color=(200,200,200), k_D=1.0, k_A=0.1, gamma=1.0, return_z=False):
    """
    Compute shading using the Fast Spheres parabolic approximation.
    """
    E_C, F_C, G_C, H_C = consts["E_C"], consts["F_C"], consts["G_C"], consts["H_C"]
    C_C, radius, max_index = consts["C_C"], consts["radius"], consts["max_index"]
    z0 = consts.get("z0", 0)

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

    brightness_norm = index[mask] / max_index
    bright = k_D * brightness_norm + k_A
    bright = np.clip(bright, 0, 1)
    bright = bright ** (1.0 / gamma)

    # Compute z-values using parabolic approximation
    z_val = (x**2 + y**2 + z0 * radius - radius**2) / radius
    # Normalise z inside mask
    z_norm = np.zeros_like(z_val, dtype=float)
    z_min = np.min(z_val[mask])
    z_max = np.max(z_val[mask])
    if z_max > z_min:
        z_norm[mask] = (z_val[mask] - z_min) / (z_max - z_min)

    image = np.zeros((index.shape[0], index.shape[1], 3), dtype=np.uint8)
    for c in range(3):
        channel = np.zeros_like(index)
        channel[mask] = base_color[c] * bright
        image[..., c] = channel.astype(np.uint8)

    if return_z:
        return image, z_norm
    return image


if __name__ == "__main__":
    consts = precompute_constants((0.3, 0.4, 1.0), radius=50, max_index=255)
    img, z = compute_shading(consts, size=200, base_color=(200,150,80), k_D=1.0, k_A=0.1, gamma=2.2, return_z=True)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Shaded Sphere")

    plt.subplot(1,2,2)
    plt.imshow(z, cmap="viridis")
    plt.axis("off")
    plt.title("Depth Map (Normalised)")

    plt.show()
