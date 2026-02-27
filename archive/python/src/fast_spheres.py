"""
fast_spheres.py — Step C1: multi-sphere rendering with a z-buffer.

This builds on:
- precompute_constants(...) from constants.py
- compute_shading(...) from shading.py

We:
- render a per-sphere disc (centre at (0, 0)) using the existing shading,
- place each disc into a larger canvas at its (cx, cy) position,
- use a z-buffer (depth map) to resolve visibility between intersecting spheres.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.constants import precompute_constants
from src.shading import compute_shading

VALID_SHADING_METHODS = {"auto", "direct", "diff", "diff_view"}


class Sphere:
    def __init__(
        self,
        center,
        radius,
        base_color=(200, 200, 200),
        light_dir=(0.3, 0.4, 1.0),
        z0=0.0,
        k_D=1.0,
        k_A=0.1,
        gamma=2.2,
    ):
        """
        centre : (cx, cy) in image coordinates (pixels)
        radius : pixel radius of the sphere
        base_color : (R,G,B) base material colour
        light_dir : (lx, ly, lz) direction TO light source
        z0 : sphere centre depth (for z-values)
        k_D, k_A : diffuse & ambient
        gamma : gamma correction
        """
        self.cx, self.cy = center
        self.radius = radius
        self.base_color = base_color
        self.light_dir = light_dir
        self.z0 = z0
        self.k_D = k_D
        self.k_A = k_A
        self.gamma = gamma


def render_spheres(
    spheres,
    img_width,
    img_height,
    background=(0, 0, 0),
    shading_method="auto",
):
    """
    Render multiple spheres into a single RGB image using a z-buffer.

    Parameters
    ----------
    spheres : list[Sphere]
        Spheres to render.
    img_width, img_height : int
        Size of the output image.
    background : (R,G,B)
        Background colour.
    shading_method : {"auto", "direct", "diff", "diff_view"}
        Per-sphere shading path selection. "auto" selects "diff_view"
        when lx=ly=0, otherwise "diff".

    Returns
    -------
    image : (H, W, 3) uint8
        Final composited image.
    z_buffer : (H, W) float
        Final depth buffer (smaller = closer).
    """
    H, W = img_height, img_width

    # Initialise canvas and z-buffer
    image = np.zeros((H, W, 3), dtype=np.float32)
    image[..., 0] = background[0]
    image[..., 1] = background[1]
    image[..., 2] = background[2]

    z_buffer = np.full((H, W), np.inf, dtype=np.float32)

    if shading_method not in VALID_SHADING_METHODS:
        raise ValueError(f"invalid shading_method={shading_method!r}")

    for sph in spheres:
        r = sph.radius

        # Precompute constants for this sphere
        consts = precompute_constants(
            light_dir=sph.light_dir,
            radius=r,
            max_index=255,
        )
        # Pass additional parameters (z0 etc.) via dict
        consts["z0"] = sph.z0

        # View-direction optimisation: radial symmetry when lx=ly=0.
        resolved_method = shading_method
        if resolved_method == "auto":
            resolved_method = (
                "diff_view"
                if np.isclose(consts["F_C"], 0.0) and np.isclose(consts["G_C"], 0.0)
                else "diff"
            )

        # Render a local disc centred at (0,0) with our existing shading code
        # This returns:
        #   img_local : (2r, 2r, 3) uint8 (approximately)
        #   z_local   : (2r, 2r) absolute depth values (smaller = nearer)
        #   mask_local: valid sphere coverage mask
        img_local, z_local, mask_local = compute_shading(
            consts,
            size=None,
            base_color=sph.base_color,
            k_D=sph.k_D,
            k_A=sph.k_A,
            gamma=sph.gamma,
            return_z=True,
            method=resolved_method,
        )

        h_local, w_local = img_local.shape[:2]

        # Compute placement of local disc into global image
        cx, cy = sph.cx, sph.cy
        x0 = int(cx - w_local // 2)
        y0 = int(cy - h_local // 2)

        # Iterate over the local disc and paste into canvas using z-buffer
        for j in range(h_local):
            gy = y0 + j
            if gy < 0 or gy >= H:
                continue

            for i in range(w_local):
                gx = x0 + i
                if gx < 0 or gx >= W:
                    continue

                # Use the geometric disc mask instead of pixel colour.
                if not mask_local[j, i]:
                    continue

                z = float(z_local[j, i])

                # Smaller z is closer to the viewer.
                if z < z_buffer[gy, gx]:
                    z_buffer[gy, gx] = z
                    image[gy, gx, :] = img_local[j, i, :]

    # Convert back to uint8
    return image.astype(np.uint8), z_buffer


def demo_scene(width=400, height=300, shading_method="auto", show=True):
    """
    Simple demo: render a few overlapping spheres.
    """
    spheres = [
        Sphere(center=(140, 160), radius=60,
               base_color=(220, 80, 80),
               light_dir=(0.3, 0.4, 1.0),
               z0=-10.0),
        Sphere(center=(220, 140), radius=70,
               base_color=(80, 200, 120),
               light_dir=(0.3, 0.4, 1.0),
               z0=-5.0),
        Sphere(center=(260, 190), radius=50,
               base_color=(80, 120, 220),
               light_dir=(0.3, 0.4, 1.0),
               z0=-2.0),
    ]

    img, zbuf = render_spheres(
        spheres,
        img_width=width,
        img_height=height,
        shading_method=shading_method,
    )

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Fast Spheres — Multi-sphere RGB + z-buffer")

    plt.subplot(1, 2, 2)
    # Show z-buffer where not inf
    z_vis = np.copy(zbuf)
    mask = np.isfinite(z_vis)
    if np.any(mask):
        z_min = z_vis[mask].min()
        z_max = z_vis[mask].max()
        if z_max > z_min:
            z_vis[mask] = (z_vis[mask] - z_min) / (z_max - z_min)
        else:
            z_vis[mask] = 0.0
    z_vis[~mask] = 1.0  # background far
    plt.imshow(z_vis, cmap="viridis")
    plt.axis("off")
    plt.title("z-buffer (normalised)")

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()

    return img, zbuf


if __name__ == "__main__":
    demo_scene()
