"""Small CLI wrapper for demo rendering and benchmark presets."""

import argparse

from src.benchmark import benchmark_scene_counts, benchmark_shading_methods
from src.fast_spheres import VALID_SHADING_METHODS, demo_scene


def _parse_args():
    parser = argparse.ArgumentParser(prog="fast_spheres")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Render the demo scene")
    demo_parser.add_argument(
        "--method",
        choices=sorted(VALID_SHADING_METHODS),
        default="auto",
        help="Shading method override for demo render",
    )
    demo_parser.add_argument("--width", type=int, default=400)
    demo_parser.add_argument("--height", type=int, default=300)
    demo_parser.add_argument("--no-show", action="store_true", help="Skip interactive plot display")

    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark presets")
    bench_parser.add_argument(
        "--preset",
        choices=("quick", "full"),
        default="quick",
        help="Benchmark preset size",
    )
    bench_parser.add_argument(
        "--method",
        choices=sorted(VALID_SHADING_METHODS),
        default="auto",
        help="Method override for scene benchmark; auto runs adaptive selection",
    )

    return parser.parse_args()


PRESETS = {
    "quick": {
        "radii": (16, 32),
        "shading_repeats": 3,
        "counts": (25, 75),
        "scene_repeats": 2,
        "width": 640,
        "height": 480,
    },
    "full": {
        "radii": (16, 32, 64, 96),
        "shading_repeats": 8,
        "counts": (25, 75, 150),
        "scene_repeats": 4,
        "width": 640,
        "height": 480,
    },
}


def main():
    args = _parse_args()

    if args.command == "demo":
        demo_scene(
            width=args.width,
            height=args.height,
            shading_method=args.method,
            show=not args.no_show,
        )
        return

    config = PRESETS[args.preset]
    methods = None if args.method == "auto" else (args.method,)

    benchmark_shading_methods(
        radii=config["radii"],
        repeats=config["shading_repeats"],
        methods=methods,
    )
    print()
    benchmark_scene_counts(
        counts=config["counts"],
        repeats=config["scene_repeats"],
        width=config["width"],
        height=config["height"],
        shading_method=args.method,
    )


if __name__ == "__main__":
    main()
