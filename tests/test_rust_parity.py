import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.fast_spheres import Sphere, render_spheres


def _load_ppm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if not data.startswith(b"P6\n"):
        raise ValueError("unsupported PPM format")

    i = 3

    def read_token() -> bytes:
        nonlocal i
        while i < len(data) and data[i] in b" \t\r\n":
            i += 1
        if i < len(data) and data[i] == ord("#"):
            while i < len(data) and data[i] != ord("\n"):
                i += 1
            return read_token()
        j = i
        while j < len(data) and data[j] not in b" \t\r\n":
            j += 1
        tok = data[i:j]
        i = j
        return tok

    width = int(read_token())
    height = int(read_token())
    maxval = int(read_token())
    if maxval != 255:
        raise ValueError("expected 8-bit PPM")

    while i < len(data) and data[i] in b" \t\r\n":
        i += 1

    rgb = np.frombuffer(data[i:], dtype=np.uint8)
    return rgb.reshape((height, width, 3))


def _python_render_from_scene(scene: dict):
    method = scene.get("shading_method", "auto")

    spheres = []
    for s in scene["spheres"]:
        cx, cy = s["center"]
        spheres.append(
            Sphere(
                center=(int(round(cx)), int(round(cy))),
                radius=int(round(s["radius"])),
                base_color=tuple(s["base_color"]),
                light_dir=tuple(s["light_dir"]),
                z0=float(s["z0"]),
                k_D=float(s["k_d"]),
                k_A=float(s["k_a"]),
                gamma=float(s["gamma"]),
            )
        )

    img, zbuf = render_spheres(
        spheres,
        img_width=int(scene["width"]),
        img_height=int(scene["height"]),
        background=tuple(scene["background"]),
        shading_method=method,
    )
    return img, zbuf


class TestRustParity(unittest.TestCase):
    def test_demo_scene_python_vs_rust(self):
        if shutil.which("cargo") is None:
            self.skipTest("cargo not found")

        repo_root = Path(__file__).resolve().parents[1]
        rust_dir = repo_root / "rust"
        scene_path = rust_dir / "scenes" / "demo_scene.json"

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ppm_path = td_path / "rust.ppm"
            stats_path = td_path / "rust_stats.json"

            cmd = [
                "cargo",
                "run",
                "-p",
                "fast_spheres_app",
                "--",
                str(scene_path),
                str(ppm_path),
                "--stats",
                str(stats_path),
            ]
            subprocess.run(cmd, cwd=rust_dir, check=True)

            rust_img = _load_ppm(ppm_path)
            rust_stats = json.loads(stats_path.read_text())

            scene = json.loads(scene_path.read_text())
            py_img, py_z = _python_render_from_scene(scene)

            self.assertEqual(py_img.shape, rust_img.shape)

            delta = np.abs(py_img.astype(np.int16) - rust_img.astype(np.int16))
            self.assertLessEqual(int(np.max(delta)), 2)

            py_rgb_sum = int(np.sum(py_img, dtype=np.uint64))
            self.assertEqual(py_rgb_sum, int(rust_stats["rgb_sum"]))

            finite = np.isfinite(py_z)
            py_count = int(np.count_nonzero(finite))
            self.assertEqual(py_count, int(rust_stats["z_finite_count"]))

            if py_count > 0:
                py_min = float(np.min(py_z[finite]))
                py_max = float(np.max(py_z[finite]))
                py_mean = float(np.mean(py_z[finite]))
                self.assertAlmostEqual(py_min, float(rust_stats["z_min"]), places=5)
                self.assertAlmostEqual(py_max, float(rust_stats["z_max"]), places=5)
                self.assertAlmostEqual(py_mean, float(rust_stats["z_mean"]), places=5)


if __name__ == "__main__":
    unittest.main()
