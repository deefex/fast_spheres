use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

use fast_spheres_core::{render_scene, Scene};
use font8x8::UnicodeFonts;
use fontdue::layout::{CoordinateSystem, Layout, LayoutSettings, TextStyle};
use image::ImageFormat;
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use serde::Serialize;

fn write_ppm(path: &Path, width: usize, height: usize, rgb: &[u8]) -> std::io::Result<()> {
    let mut data = format!("P6\n{} {}\n255\n", width, height).into_bytes();
    data.extend_from_slice(rgb);
    fs::write(path, data)
}

fn write_png(path: &Path, width: usize, height: usize, rgb: &[u8]) -> image::ImageResult<()> {
    image::save_buffer_with_format(
        path,
        rgb,
        width as u32,
        height as u32,
        image::ColorType::Rgb8,
        ImageFormat::Png,
    )
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

#[derive(Debug)]
struct CliArgs {
    input: PathBuf,
    scene2: Option<PathBuf>,
    output: PathBuf,
    stats_path: Option<PathBuf>,
    interactive: bool,
    continuous: bool,
}

struct UiFont {
    font: Option<fontdue::Font>,
    size: f32,
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

fn parse_args(args: &[String]) -> CliArgs {
    if args.len() < 2 {
        eprintln!(
            "Usage: cargo run -p fast_spheres_app -- <scene.json> [out.ppm] [--scene2 <scene2.json>] [--stats <stats.json>] [--interactive] [--continuous]"
        );
        std::process::exit(2);
    }

    let input = Path::new(&args[1]).to_path_buf();
    let mut scene2: Option<PathBuf> = None;
    let mut output = Path::new("render.ppm").to_path_buf();
    let mut stats_path: Option<PathBuf> = None;
    let mut interactive = false;
    let mut continuous = false;

    let mut i = 2usize;
    if i < args.len() && !args[i].starts_with("--") {
        output = Path::new(&args[i]).to_path_buf();
        i += 1;
    }

    while i < args.len() {
        match args[i].as_str() {
            "--stats" => {
                if i + 1 >= args.len() {
                    eprintln!("missing value for --stats");
                    std::process::exit(2);
                }
                stats_path = Some(Path::new(&args[i + 1]).to_path_buf());
                i += 2;
            }
            "--scene2" => {
                if i + 1 >= args.len() {
                    eprintln!("missing value for --scene2");
                    std::process::exit(2);
                }
                scene2 = Some(Path::new(&args[i + 1]).to_path_buf());
                i += 2;
            }
            "--interactive" => {
                interactive = true;
                i += 1;
            }
            "--continuous" => {
                continuous = true;
                i += 1;
            }
            unknown => {
                eprintln!("unknown argument: {}", unknown);
                std::process::exit(2);
            }
        }
    }

    CliArgs {
        input,
        scene2,
        output,
        stats_path,
        interactive,
        continuous,
    }
}

fn rgb_to_u32_buffer_inplace(rgb: &[u8], out: &mut Vec<u32>) {
    let needed = rgb.len() / 3;
    if out.len() != needed {
        out.resize(needed, 0);
    }
    for (i, px) in rgb.chunks_exact(3).enumerate() {
        out[i] = ((px[0] as u32) << 16) | ((px[1] as u32) << 8) | (px[2] as u32);
    }
}

fn load_ui_font() -> UiFont {
    let mut paths: Vec<PathBuf> = Vec::new();
    if let Ok(custom) = env::var("FAST_SPHERES_FONT") {
        paths.push(PathBuf::from(custom));
    }
    paths.extend([
        PathBuf::from("/System/Library/Fonts/Supplemental/Arial.ttf"),
        PathBuf::from("/System/Library/Fonts/Supplemental/Helvetica.ttf"),
        PathBuf::from("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        PathBuf::from("C:\\Windows\\Fonts\\arial.ttf"),
    ]);

    for p in paths {
        if let Ok(bytes) = fs::read(&p) {
            if let Ok(font) = fontdue::Font::from_bytes(bytes, fontdue::FontSettings::default()) {
                println!("ui font: {}", p.display());
                return UiFont {
                    font: Some(font),
                    size: 13.0,
                };
            }
        }
    }

    eprintln!("ui font: no system TTF found; falling back to bitmap font");
    UiFont {
        font: None,
        size: 13.0,
    }
}

fn blend_pixel(dst: u32, src: u32, alpha: f32) -> u32 {
    let a = alpha.clamp(0.0, 1.0);
    let dr = ((dst >> 16) & 0xff) as f32;
    let dg = ((dst >> 8) & 0xff) as f32;
    let db = (dst & 0xff) as f32;
    let sr = ((src >> 16) & 0xff) as f32;
    let sg = ((src >> 8) & 0xff) as f32;
    let sb = (src & 0xff) as f32;
    let r = (sr * a + dr * (1.0 - a)) as u32;
    let g = (sg * a + dg * (1.0 - a)) as u32;
    let b = (sb * a + db * (1.0 - a)) as u32;
    (r << 16) | (g << 8) | b
}

fn normalise(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n <= f32::EPSILON {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

fn to_yaw_pitch(v: [f32; 3]) -> (f32, f32) {
    let n = normalise(v);
    let yaw = n[0].atan2(n[2]);
    let pitch = n[1].asin();
    (yaw, pitch)
}

fn from_yaw_pitch(yaw: f32, pitch: f32) -> [f32; 3] {
    let cp = pitch.cos();
    normalise([cp * yaw.sin(), pitch.sin(), cp * yaw.cos()])
}

fn run_batch(scene: &Scene, output: &Path, stats_path: Option<&Path>) {
    let result = render_scene(scene);
    write_ppm(output, result.width, result.height, &result.rgb)
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

fn draw_text(
    buf: &mut [u32],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    text: &str,
    color: u32,
    ui_font: &UiFont,
) {
    if let Some(font) = &ui_font.font {
        let mut layout = Layout::new(CoordinateSystem::PositiveYDown);
        layout.reset(&LayoutSettings {
            x: x as f32,
            y: y as f32,
            ..LayoutSettings::default()
        });
        layout.append(&[font], &TextStyle::new(text, ui_font.size, 0));

        for glyph in layout.glyphs() {
            let (metrics, bitmap) = font.rasterize_config(glyph.key);
            for gy in 0..metrics.height {
                for gx in 0..metrics.width {
                    let alpha = bitmap[gy * metrics.width + gx] as f32 / 255.0;
                    if alpha <= 0.01 {
                        continue;
                    }
                    let px = glyph.x as i32 + gx as i32;
                    let py = glyph.y as i32 + gy as i32;
                    if px >= 0 && py >= 0 && px < width as i32 && py < height as i32 {
                        let idx = py as usize * width + px as usize;
                        buf[idx] = blend_pixel(buf[idx], color, alpha);
                    }
                }
            }
        }
        return;
    }

    // Fallback bitmap font.
    let mut pen_x = x;
    for ch in text.chars() {
        if let Some(glyph) = font8x8::BASIC_FONTS.get(ch) {
            for (gy, row_bits) in glyph.iter().enumerate() {
                for gx in 0..8 {
                    if (row_bits >> gx) & 1 == 1 {
                        let px = pen_x + gx as i32;
                        let py = y + gy as i32;
                        if px >= 0 && py >= 0 && px < width as i32 && py < height as i32 {
                            let idx = py as usize * width + px as usize;
                            buf[idx] = color;
                        }
                    }
                }
            }
        }
        pen_x += 8;
    }
}

fn draw_help_overlay(
    buf: &mut [u32],
    width: usize,
    height: usize,
    has_scene2: bool,
    active_scene: u8,
    parallel: bool,
    continuous: bool,
    ui_font: &UiFont,
) {
    let panel_w = 430i32;
    let panel_h = 58i32;
    let panel_x = 10i32;
    let panel_y = height as i32 - panel_h - 10;

    for py in panel_y..(panel_y + panel_h) {
        for px in panel_x..(panel_x + panel_w) {
            if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
                continue;
            }
            let idx = py as usize * width + px as usize;
            let c = buf[idx];
            let r = ((c >> 16) & 0xff) as u8;
            let g = ((c >> 8) & 0xff) as u8;
            let b = (c & 0xff) as u8;
            let dr = (r as f32 * 0.70) as u8;
            let dg = (g as f32 * 0.70) as u8;
            let db = (b as f32 * 0.70) as u8;
            buf[idx] = ((dr as u32) << 16) | ((dg as u32) << 8) | (db as u32);
        }
    }

    let white = 0x00ff_ffffu32;
    let cyan = 0x0000_ffffu32;
    draw_text(
        buf,
        width,
        height,
        panel_x + 8,
        panel_y + 8,
        "H help  C redraw  T renderer  P shot  R reset  ESC quit",
        white,
        ui_font,
    );
    draw_text(
        buf,
        width,
        height,
        panel_x + 8,
        panel_y + 24,
        "Arrows/WASD or drag mouse: light direction",
        white,
        ui_font,
    );
    let status = format!(
        "1/2 scene: {} | scene={} renderer={} redraw={}",
        if has_scene2 { "available" } else { "none" },
        active_scene,
        if parallel { "parallel" } else { "sequential" },
        if continuous {
            "continuous"
        } else {
            "on-change"
        }
    );
    draw_text(
        buf,
        width,
        height,
        panel_x + 8,
        panel_y + 40,
        &status,
        cyan,
        ui_font,
    );
}

fn run_interactive(scene: &Scene, scene2: Option<Scene>, mut continuous: bool) {
    let mut window = Window::new(
        "Fast Spheres (interactive light)",
        scene.width as usize,
        scene.height as usize,
        WindowOptions::default(),
    )
    .expect("failed to create window");

    window.set_target_fps(60);

    let initial_light = scene
        .spheres
        .first()
        .map(|s| s.light_dir)
        .unwrap_or([0.3, 0.4, 1.0]);
    let (mut yaw, mut pitch) = to_yaw_pitch(initial_light);
    let (base_yaw, base_pitch) = (yaw, pitch);

    let mut frame_count: u64 = 0;
    let mut last_fps_t = Instant::now();
    let mut last_mouse_pos: Option<(f32, f32)> = None;
    let mut prev_toggle_state = false;
    let mut prev_snapshot_state = false;
    let mut prev_parallel_toggle_state = false;
    let mut prev_help_toggle_state = false;
    let mut prev_scene1_state = false;
    let mut prev_scene2_state = false;
    let mut show_help = false;
    let mut fps = 0.0f64;
    let mut current_rgb: Vec<u8> = Vec::new();
    let mut display_buf: Vec<u32> = Vec::new();
    let mut current_width = scene.width as usize;
    let mut current_height = scene.height as usize;
    let mut frame_scene = scene.clone();
    let scene1 = scene.clone();
    let scene2 = scene2;
    let mut active_scene: u8 = 1;
    let ui_font = load_ui_font();

    println!(
        "interactive start: redraw_mode={} renderer={}",
        if continuous {
            "continuous"
        } else {
            "on-change"
        },
        if frame_scene.parallel {
            "parallel"
        } else {
            "sequential"
        }
    );

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut changed = false;
        let step = 0.03f32;

        if window.is_key_down(Key::Left) || window.is_key_down(Key::A) {
            yaw -= step;
            changed = true;
        }
        if window.is_key_down(Key::Right) || window.is_key_down(Key::D) {
            yaw += step;
            changed = true;
        }
        if window.is_key_down(Key::Up) || window.is_key_down(Key::W) {
            pitch += step;
            changed = true;
        }
        if window.is_key_down(Key::Down) || window.is_key_down(Key::S) {
            pitch -= step;
            changed = true;
        }
        if window.is_key_down(Key::R) {
            yaw = base_yaw;
            pitch = base_pitch;
            changed = true;
        }

        let toggle_state = window.is_key_down(Key::C);
        if toggle_state && !prev_toggle_state {
            continuous = !continuous;
            changed = true;
        }
        prev_toggle_state = toggle_state;

        let parallel_toggle_state = window.is_key_down(Key::T);
        if parallel_toggle_state && !prev_parallel_toggle_state {
            frame_scene.parallel = !frame_scene.parallel;
            changed = true;
        }
        prev_parallel_toggle_state = parallel_toggle_state;

        let help_toggle_state = window.is_key_down(Key::H);
        if help_toggle_state && !prev_help_toggle_state {
            show_help = !show_help;
            changed = true;
        }
        prev_help_toggle_state = help_toggle_state;

        let scene1_state = window.is_key_down(Key::Key1);
        if scene1_state && !prev_scene1_state && active_scene != 1 {
            if scene1.width == frame_scene.width && scene1.height == frame_scene.height {
                let prev_parallel = frame_scene.parallel;
                frame_scene = scene1.clone();
                frame_scene.parallel = prev_parallel;
                active_scene = 1;
                changed = true;
            } else {
                eprintln!("scene 1 dimensions differ from active window; cannot switch");
            }
        }
        prev_scene1_state = scene1_state;

        let scene2_state = window.is_key_down(Key::Key2);
        if scene2_state && !prev_scene2_state && active_scene != 2 {
            if let Some(s2) = &scene2 {
                if s2.width == frame_scene.width && s2.height == frame_scene.height {
                    let prev_parallel = frame_scene.parallel;
                    frame_scene = s2.clone();
                    frame_scene.parallel = prev_parallel;
                    active_scene = 2;
                    changed = true;
                } else {
                    eprintln!("scene 2 dimensions differ from active window; cannot switch");
                }
            }
        }
        prev_scene2_state = scene2_state;

        let snapshot_state = window.is_key_down(Key::P);
        if snapshot_state && !prev_snapshot_state && !current_rgb.is_empty() {
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let filename = format!("snapshot_{}.png", ts);
            let path = Path::new(&filename);
            match write_png(path, current_width, current_height, &current_rgb) {
                Ok(_) => println!("saved snapshot {}", path.display()),
                Err(e) => eprintln!("failed to save snapshot {}: {}", path.display(), e),
            }
        }
        prev_snapshot_state = snapshot_state;

        if window.get_mouse_down(MouseButton::Left) {
            if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Pass) {
                if let Some((px, py)) = last_mouse_pos {
                    yaw += (mx - px) * 0.01;
                    pitch -= (my - py) * 0.01;
                    changed = true;
                }
                last_mouse_pos = Some((mx, my));
            }
        } else {
            last_mouse_pos = None;
        }

        pitch = pitch.clamp(-1.4, 1.4);

        // Draw at least once, then redraw on input change or always in continuous mode.
        if frame_count == 0 || changed || continuous {
            let light = from_yaw_pitch(yaw, pitch);
            for s in &mut frame_scene.spheres {
                s.light_dir = light;
            }

            let result = render_scene(&frame_scene);
            current_width = result.width;
            current_height = result.height;
            current_rgb = result.rgb;
            rgb_to_u32_buffer_inplace(&current_rgb, &mut display_buf);
            if show_help {
                draw_help_overlay(
                    &mut display_buf,
                    current_width,
                    current_height,
                    scene2.is_some(),
                    active_scene,
                    frame_scene.parallel,
                    continuous,
                    &ui_font,
                );
            }
            window
                .update_with_buffer(&display_buf, current_width, current_height)
                .expect("failed to update window buffer");
        } else {
            window.update();
        }

        frame_count += 1;
        let elapsed = last_fps_t.elapsed();
        if elapsed >= Duration::from_secs(1) {
            fps = frame_count as f64 / elapsed.as_secs_f64();
            let light = from_yaw_pitch(yaw, pitch);
            println!(
                "fps={:.1} light=({:.3}, {:.3}, {:.3})",
                fps, light[0], light[1], light[2]
            );
            frame_count = 0;
            last_fps_t = Instant::now();
        }

        let light = from_yaw_pitch(yaw, pitch);
        let mode = if continuous {
            "continuous"
        } else {
            "on-change"
        };
        let parallel_mode = if frame_scene.parallel {
            "parallel"
        } else {
            "sequential"
        };
        let scene_info = if scene2.is_some() {
            format!("scene={}", active_scene)
        } else {
            "scene=1".to_string()
        };
        window.set_title(&format!(
            "Fast Spheres | fps={:.1} | {} | mode={} | renderer={} | light=({:.3}, {:.3}, {:.3}) | H help C redraw T renderer 1/2 scenes P snapshot R reset Esc quit",
            fps, scene_info, mode, parallel_mode, light[0], light[1], light[2]
        ));
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let cli = parse_args(&args);

    let json1 = fs::read_to_string(&cli.input).expect("failed to read input scene json");
    let scene: Scene = serde_json::from_str(&json1).expect("failed to parse scene json");
    let scene2 = if let Some(path) = &cli.scene2 {
        let json2 = fs::read_to_string(path).expect("failed to read scene2 json");
        Some(serde_json::from_str::<Scene>(&json2).expect("failed to parse scene2 json"))
    } else {
        None
    };

    if cli.interactive {
        run_interactive(&scene, scene2, cli.continuous);
    } else {
        run_batch(&scene, &cli.output, cli.stats_path.as_deref());
    }
}
