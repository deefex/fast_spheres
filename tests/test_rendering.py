import unittest

import numpy as np

from src.constants import precompute_constants
from src.fast_spheres import Sphere, render_spheres
from src.shading import compute_shading


class TestShadingParity(unittest.TestCase):
    def test_direct_vs_diff_exact(self):
        cases = [
            ((0.3, 0.4, 1.0), 12),
            ((-0.2, 0.5, 1.0), 18),
            ((0.1, -0.3, 1.0), 25),
        ]

        for light_dir, radius in cases:
            with self.subTest(light_dir=light_dir, radius=radius):
                consts = precompute_constants(light_dir, radius=radius, max_index=255)
                consts["z0"] = -6.0

                img_d, z_d, mask_d = compute_shading(consts, return_z=True, method="direct")
                img_x, z_x, mask_x = compute_shading(consts, return_z=True, method="diff")

                self.assertTrue(np.array_equal(mask_d, mask_x))
                img_delta = np.abs(img_d.astype(np.int16) - img_x.astype(np.int16))
                self.assertLessEqual(int(np.max(img_delta)), 1)
                self.assertTrue(np.array_equal(z_d[mask_d], z_x[mask_x]))

    def test_direct_vs_diff_view_tolerance(self):
        consts = precompute_constants((0.0, 0.0, -1.0), radius=40, max_index=255)
        consts["z0"] = -9.0

        img_d, z_d, mask_d = compute_shading(consts, return_z=True, method="direct")
        img_v, z_v, mask_v = compute_shading(consts, return_z=True, method="diff_view")

        self.assertTrue(np.array_equal(mask_d, mask_v))
        self.assertTrue(np.array_equal(z_d[mask_d], z_v[mask_v]))

        img_delta = np.abs(img_d.astype(np.int16) - img_v.astype(np.int16))
        self.assertLessEqual(int(np.max(img_delta)), 1)


class TestZBufferOrdering(unittest.TestCase):
    def test_nearer_sphere_wins_overlap(self):
        width, height = 120, 120
        center = (60, 60)

        far = Sphere(
            center=center,
            radius=28,
            base_color=(30, 90, 220),
            light_dir=(0.3, 0.4, 1.0),
            z0=2.0,
        )
        near = Sphere(
            center=center,
            radius=20,
            base_color=(220, 80, 60),
            light_dir=(0.3, 0.4, 1.0),
            z0=-12.0,
        )

        img_far_only, _ = render_spheres([far], img_width=width, img_height=height)
        img_near_only, _ = render_spheres([near], img_width=width, img_height=height)

        img_scene, z_scene = render_spheres([far, near], img_width=width, img_height=height)

        cx, cy = center
        near_px = img_near_only[cy, cx, :]
        far_px = img_far_only[cy, cx, :]
        scene_px = img_scene[cy, cx, :]

        self.assertTrue(np.array_equal(scene_px, near_px))
        self.assertFalse(np.array_equal(scene_px, far_px))
        self.assertTrue(np.isfinite(z_scene[cy, cx]))


if __name__ == "__main__":
    unittest.main()
