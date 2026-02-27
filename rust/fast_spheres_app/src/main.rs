use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use fast_spheres_core::{render_scene, Scene};
use serde::Serialize;

fn write_ppm(path: &Path, width: usize, height: usize, rgb: &[u8]) -> std::io::Result<()> {
    let mut data = format!("P6\n{} {}\n255\n", width, height).into_bytes();
    data.extend_from_slice(rgb);
    fs::write(path, data)
}

#[derive(Debug, Serialize)]
struct RenderStats {
    width: usize,
    height: usize,
    rgb_sum: u64,
    z_finite_count: usize,
    z_min: Option<f32>,
    z_max: Option<f32>,
    z_mean: Option<f32>,
}

fn compute_stats(width: usize, height: usize, rgb: &[u8], z_buffer: &[f32]) -> RenderStats {
    let rgb_sum = rgb.iter().map(|v| *v as u64).sum::<u64>();

    let mut finite_count = 0usize;
    let mut z_min = f32::INFINITY;
    let mut z_max = f32::NEG_INFINITY;
    let mut z_sum = 0.0f64;

    for z in z_buffer {
        if z.is_finite() {
            finite_count += 1;
            z_min = z_min.min(*z);
            z_max = z_max.max(*z);
            z_sum += *z as f64;
        }
    }

    let (z_min_opt, z_max_opt, z_mean_opt) = if finite_count > 0 {
        (
            Some(z_min),
            Some(z_max),
            Some((z_sum / finite_count as f64) as f32),
        )
    } else {
        (None, None, None)
    };

    RenderStats {
        width,
        height,
        rgb_sum,
        z_finite_count: finite_count,
        z_min: z_min_opt,
        z_max: z_max_opt,
        z_mean: z_mean_opt,
    }
}

fn parse_args(args: &[String]) -> (PathBuf, PathBuf, Option<PathBuf>) {
    if args.len() < 2 {
        eprintln!(
            "Usage: cargo run -p fast_spheres_app -- <scene.json> [out.ppm] [--stats <stats.json>]"
        );
        std::process::exit(2);
    }

    let input = Path::new(&args[1]).to_path_buf();
    let mut output = Path::new("render.ppm").to_path_buf();
    let mut stats_path: Option<PathBuf> = None;

    let mut i = 2usize;
    if i < args.len() && args[i] != "--stats" {
        output = Path::new(&args[i]).to_path_buf();
        i += 1;
    }

    while i < args.len() {
        if args[i] == "--stats" {
            if i + 1 >= args.len() {
                eprintln!("missing value for --stats");
                std::process::exit(2);
            }
            stats_path = Some(Path::new(&args[i + 1]).to_path_buf());
            i += 2;
            continue;
        }

        eprintln!("unknown argument: {}", args[i]);
        std::process::exit(2);
    }

    (input, output, stats_path)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let (input, output, stats_path) = parse_args(&args);

    let json = fs::read_to_string(input).expect("failed to read input scene json");
    let scene: Scene = serde_json::from_str(&json).expect("failed to parse scene json");

    let result = render_scene(&scene);
    write_ppm(&output, result.width, result.height, &result.rgb)
        .expect("failed to write output image");

    if let Some(path) = stats_path {
        let stats = compute_stats(result.width, result.height, &result.rgb, &result.z_buffer);
        let stats_json =
            serde_json::to_string_pretty(&stats).expect("failed to serialise stats json");
        fs::write(path, stats_json).expect("failed to write stats json");
    }

    println!(
        "Rendered {}x{} image to {}",
        result.width,
        result.height,
        output.display()
    );
}
