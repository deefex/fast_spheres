# fast_spheres

Python recreation of the Fast Spheres rendering approach from the original
university project/paper work: Lambert-style shaded sphere images using a
parabolic surface approximation and differencing-based raster paths.

## What this repo currently does

- Renders multiple intersecting spheres into a single image.
- Uses a z-buffer for visibility between overlapping spheres.
- Supports three per-sphere shading paths:
  - `direct`: dense vectorized reference implementation
  - `diff`: Bresenham-bounded row filling + differencing updates
  - `diff_view`: optimized view-direction variant using octant symmetry
- Includes parity tests and benchmark tooling.

## Quick start

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main entrypoint

Run the demo scene:

```bash
python -m src.cli demo
```

Useful demo options:

```bash
python -m src.cli demo --method auto
python -m src.cli demo --method diff
python -m src.cli demo --method diff_view
python -m src.cli demo --no-show --width 640 --height 480
```

`--method` values:
- `auto`: choose `diff_view` for view-direction lighting, else `diff`
- `direct`: force direct/vectorized shading
- `diff`: force differencing path
- `diff_view`: force view-direction octant-symmetry path

## Benchmarks

Quick preset:

```bash
python -m src.cli benchmark --preset quick
```

Full preset:

```bash
python -m src.cli benchmark --preset full
```

Force a specific method in scene benchmarks:

```bash
python -m src.cli benchmark --preset quick --method diff
```

## Rust workspace (scaffold)

A Rust workspace now exists at `rust/` for the next phase:

- `rust/fast_spheres_core`: renderer core library
- `rust/fast_spheres_app`: small binary that loads a JSON scene and writes a PPM image
- `rust/scenes/demo_scene.json`: starter scene

Run it:

```bash
cd rust
cargo run -p fast_spheres_app -- scenes/demo_scene.json out.ppm
```

This is an initial scaffold and currently uses a straightforward baseline renderer.
`Scene` JSON now supports `"shading_method"` with values:
- `"auto"`
- `"direct"`
- `"diff"`
- `"diff_view"`

## Tests

Run regression tests:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Current tests cover:
- parity between `direct` and `diff`
- parity/tolerance for `direct` vs `diff_view` in view-direction lighting
- z-buffer overlap ordering (nearer sphere wins)
- cross-language parity check (Python vs Rust on shared demo scene)

Convenience make targets:

```bash
make parity
make test-rust
make test-py
make test
make fmt
make fmt-check
```

## File map

- `src/cli.py`: user entrypoint (`demo`, `benchmark`)
- `src/fast_spheres.py`: scene rendering and z-buffer composition
- `src/shading.py`: per-sphere shading implementations
- `src/constants.py`: precomputed constants for shading equations
- `src/benchmark.py`: benchmark helpers
- `tests/test_rendering.py`: rendering and z-buffer regression tests
- `experiments/diff_shading.py`: legacy analysis/validation script
