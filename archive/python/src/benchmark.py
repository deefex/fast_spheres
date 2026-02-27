"""Benchmark helpers for Fast Spheres shading and scene rendering."""

from __future__ import annotations

import time
import numpy as np

from src.constants import precompute_constants
from src.fast_spheres import Sphere, render_spheres
from src.shading import compute_shading


def _time_call(fn, repeats):
    durations = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - t0)
    return np.asarray(durations, dtype=np.float64)


def benchmark_shading_methods(radii=(16, 32, 64, 96), repeats=8, methods=None):
    """Benchmark direct, diff, and diff_view per-sphere shading paths."""
    methods = ("direct", "diff", "diff_view") if methods is None else tuple(methods)
    lights = {
        "tilted": (0.3, 0.4, 1.0),
        "view": (0.0, 0.0, -1.0),
    }

    print("=== Per-Sphere Shading Benchmark ===")
    print(f"repeats per case: {repeats}")

    for radius in radii:
        for light_name, light_dir in lights.items():
            consts = precompute_constants(light_dir, radius=radius, max_index=255)
            consts["z0"] = -8.0
            print(f"radius={radius:3d} | light={light_name:6s}")

            for method in methods:
                durations = _time_call(
                    lambda m=method: compute_shading(
                        consts,
                        base_color=(180, 180, 200),
                        k_D=1.0,
                        k_A=0.1,
                        gamma=2.2,
                        return_z=True,
                        method=m,
                    ),
                    repeats,
                )
                mean_ms = 1000.0 * float(np.mean(durations))
                min_ms = 1000.0 * float(np.min(durations))
                print(f"  {method:9s} mean={mean_ms:8.3f} ms  min={min_ms:8.3f} ms")



def _build_scene(sphere_count, width, height, light_dir, seed=7):
    rng = np.random.default_rng(seed)
    spheres = []
    for _ in range(sphere_count):
        r = int(rng.integers(8, 22))
        cx = int(rng.integers(r, width - r))
        cy = int(rng.integers(r, height - r))
        z0 = float(rng.uniform(-20.0, 5.0))
        color = tuple(int(v) for v in rng.integers(40, 240, size=3))
        spheres.append(
            Sphere(
                center=(cx, cy),
                radius=r,
                base_color=color,
                light_dir=light_dir,
                z0=z0,
                k_D=1.0,
                k_A=0.1,
                gamma=2.2,
            )
        )
    return spheres


def benchmark_scene_counts(
    counts=(25, 75, 150),
    repeats=4,
    width=640,
    height=480,
    shading_method="auto",
):
    """Benchmark full scene rendering as sphere count increases."""
    print("=== Multi-Sphere Scene Benchmark ===")
    print(
        f"canvas={width}x{height}, repeats per case: {repeats}, "
        f"method={shading_method}"
    )

    cases = {
        "tilted": (0.3, 0.4, 1.0),
        "view": (0.0, 0.0, -1.0),
    }

    for name, light_dir in cases.items():
        print(f"light={name}")
        for count in counts:
            spheres = _build_scene(count, width, height, light_dir=light_dir, seed=11 + count)
            durations = _time_call(
                lambda s=spheres: render_spheres(
                    s,
                    img_width=width,
                    img_height=height,
                    shading_method=shading_method,
                ),
                repeats,
            )
            mean_ms = 1000.0 * float(np.mean(durations))
            min_ms = 1000.0 * float(np.min(durations))
            print(f"  spheres={count:4d} mean={mean_ms:8.3f} ms  min={min_ms:8.3f} ms")


if __name__ == "__main__":
    benchmark_shading_methods()
    print()
    benchmark_scene_counts()
