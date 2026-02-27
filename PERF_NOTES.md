# Performance Notes

Track interactive FPS over time for reproducible comparisons.

## How to measure

1. Run each command below.
2. Let it settle for a few seconds.
3. Record the FPS shown in the window title/terminal.

Commands:
- `cd /Users/deefex/github/fast_spheres && make rust-interactive`
- `cd /Users/deefex/github/fast_spheres/rust && cargo run -p fast_spheres_app -- scenes/demo_scene.json --interactive --continuous`
- `cd /Users/deefex/github/fast_spheres && make rust-interactive-dense`
- `cd /Users/deefex/github/fast_spheres/rust && cargo run -p fast_spheres_app -- scenes/demo_dense.json --interactive`

## Log

| Date (UTC) | Machine | Scene | Mode | FPS | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-02-27 18:00 UTC | Mac Mini M4 | demo_scene | on-change | 56.2 | includes light movement |
| 2026-02-27 18:03 UTC | Mac Mini M4 | demo_scene | continuous | 56.5 | includes light movement |
| 2026-02-27 18:05 UTC | Mac Mini M4 | demo_dense | continuous | 30.5 | includes light movement |
| 2026-02-27 18:06 UTC | Mac Mini M4 | demo_dense | on-change | 43.7 | high variance during active light movement |
| 2026-02-27 18:10 UTC | Mac Mini M4 | demo_scene | on-change | 56.3 | includes light movement (post-optimization) |
| 2026-02-27 18:11 UTC | Mac Mini M4 | demo_scene | continuous | 56.5 | includes light movement (post-optimization) |
| 2026-02-27 18:12 UTC | Mac Mini M4 | demo_dense | continuous | 30.6 | includes light movement (post-optimization) |
| 2026-02-27 18:13 UTC | Mac Mini M4 | demo_dense | on-change | 41.6 | high variance during active light movement (post-optimization) |
| 2026-02-27 18:19 UTC | Mac Mini M4 | demo_dense_parallel | continuous | 35.0 | parallel=true; includes light movement |
| 2026-02-27 18:27 UTC | Mac Mini M4 | demo_dense_parallel | continuous | 35.3 | renderer=parallel, post-toggle session |
| 2026-02-27 18:28 UTC | Mac Mini M4 | demo_dense_parallel | continuous | 28.7 | renderer=sequential via T toggle |

