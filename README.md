# fast_spheres
[![CI](https://github.com/deefex/fast_spheres/actions/workflows/ci.yml/badge.svg)](https://github.com/deefex/fast_spheres/actions/workflows/ci.yml)

## Fast Spheres

'Fast Spheres' was my Senior Honours project way back in the day at the University of Glasgow for Professor John W. Patterson, who published a [paper on the topic](https://diglib.eg.org/items/23326b6a-bccd-4365-9696-33895aae53ab) for EuroGraphics, and was kind enough to credit me in it for my efforts in brining the algorithm to life.

Abstract: 
'A new method for generating sphere-like images, using parabolic surfaces delimited by Bresenham s circle- generation algorithm, is presented. In many cases the resultant images are indistinguishable from spheres illuminated from a given single direction. The main form of the algorithm uses first and second integer differences to minimise computation and uses typically not more than one floating-point calculation per sphere. Two variants of the algorithm are given, one optimized for the special case of the light-source being behind the view-point, and one in which values are calculated for a z-buffer hidden-surface algorithm at the same time as the pixel values. The difference formulae can be exploited by differencing hardware or digital signal processors although very little arithmetic, typically five low-weight integer operations, including address calculation operations, is required per pixel on a conventional architecture.'

The original project was implemented in C on a Sun SPARCstation, where minimizing per-pixel arithmetic and memory overhead mattered heavily, and after finding my old notes in the attic, I was curious how it would look using modern tooling. 

This repository preserves that algorithmic spirit while modernizing the implementation:

- Primary engine in Rust for interactive rendering and performance experiments.
- Archived Python implementation kept as a reference/oracle for parity checks.
- Added modern tooling (tests, CI, release artifacts, perf notes) around the original rendering ideas.

## Primary Engine (Rust)

The primary implementation is in `rust/`:

- `rust/fast_spheres_core`: rendering core (Lambert/parabolic fast-sphere variants)
- `rust/fast_spheres_app`: interactive app + file render utility
- `rust/scenes/*.json`: demo scene inputs

## Quick Start (Rust)

```bash
cd /Users/deefex/github/fast_spheres/rust
cargo run -p fast_spheres_app -- scenes/demo_scene.json --interactive --scene2 scenes/demo_scene_alt.json
```

Dense stress scene:

```bash
cd /Users/deefex/github/fast_spheres/rust
cargo run -p fast_spheres_app -- scenes/demo_dense_parallel.json --interactive --continuous --scene2 scenes/demo_dense.json
```

## Controls

- `Arrow keys` / `WASD`: move light direction
- `Left mouse drag`: rotate light direction
- `H`: toggle help overlay
- `C`: toggle redraw mode (`on-change` / `continuous`)
- `T`: toggle renderer (`sequential` / `parallel`)
- `1` / `2`: switch between primary and secondary scenes (`--scene2`)
- `P`: save PNG snapshot to `rust/snapshot_<unix_ts>.png`
- `R`: reset light
- `Esc`: quit

## Screenshots

Demo scene (clean view):

![Demo Scene Clean](images/demo_scene1.png)

Demo scene (with compact help overlay):

![Demo Scene Overlay](images/demo_scene2.png)

Dense scene (continuous + parallel):

![Dense Scene Parallel 1](images/dense_scene1.png)

Dense scene (alternate light/view):

![Dense Scene Parallel 2](images/dense_scene2.png)

## Performance Tracking

Use:

```bash
cd /Users/deefex/github/fast_spheres
make perf-note
```

Then run the interactive scenarios and fill FPS values in `PERF_NOTES.md`.

## Tests

Rust + parity tests:

```bash
cd /Users/deefex/github/fast_spheres
make test
```

## CI and Releases

GitHub Actions workflows:

- `.github/workflows/ci.yml`
  - Rust formatting/check/tests
  - Python↔Rust parity test
- `.github/workflows/release.yml`
  - Runs on tags matching `v*`
  - Builds `fast_spheres_app` on Linux/macOS/Windows
  - Uploads packaged binaries/scenes to the GitHub Release

To cut a release:

```bash
git tag v0.2.0
git push origin v0.2.0
```

## Archived Python Implementation

The original Python recreation is archived under:

- `archive/python/src`
- `archive/python/tests`
- `archive/python/requirements.txt`

Legacy commands:

```bash
cd /Users/deefex/github/fast_spheres
make legacy-test-py
make legacy-bench
make legacy-demo
```

## Common Make Targets

```bash
make demo
make bench
make rust-interactive
make rust-interactive-dense
make rust-interactive-dense-parallel
make parity
make test
make fmt
make fmt-check
```
