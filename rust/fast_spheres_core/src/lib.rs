pub mod scene;

pub use scene::{Scene, ShadingMethod, Sphere};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use std::sync::Mutex;

#[derive(Clone, Debug)]
pub struct RenderResult {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    pub z_buffer: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
struct Constants {
    e: f64,
    f: f64,
    g: f64,
    h: f64,
    k: f64,
    l: f64,
}

pub fn render_scene(scene: &Scene) -> RenderResult {
    #[cfg(feature = "parallel")]
    if scene.parallel {
        return render_scene_parallel(scene);
    }

    let width = scene.width as usize;
    let height = scene.height as usize;

    let mut rgb = vec![0u8; width * height * 3];
    let mut z_buffer = vec![f32::INFINITY; width * height];

    for px in rgb.chunks_exact_mut(3) {
        px[0] = scene.background[0];
        px[1] = scene.background[1];
        px[2] = scene.background[2];
    }

    for sphere in &scene.spheres {
        let radius = sphere.radius.round() as i32;
        if radius <= 0 {
            continue;
        }

        let consts = precompute_constants(sphere.light_dir, radius, 255);
        let method = resolve_method(scene.shading_method, consts);

        let mut writer = |gx: i32, gy: i32, z: f32, brightness: f32| {
            write_pixel_if_nearer_seq(
                gx,
                gy,
                z,
                brightness,
                sphere,
                width,
                height,
                &mut rgb,
                &mut z_buffer,
            )
        };

        render_sphere_with_method(sphere, radius, consts, method, &mut writer);
    }

    RenderResult {
        width,
        height,
        rgb,
        z_buffer,
    }
}

#[cfg(feature = "parallel")]
#[derive(Debug)]
struct RowData {
    z: Vec<f32>,
    rgb: Vec<u8>,
}

#[cfg(feature = "parallel")]
fn render_scene_parallel(scene: &Scene) -> RenderResult {
    let width = scene.width as usize;
    let height = scene.height as usize;

    let mut rows = Vec::with_capacity(height);
    for _ in 0..height {
        let mut row = RowData {
            z: vec![f32::INFINITY; width],
            rgb: vec![0u8; width * 3],
        };
        for px in row.rgb.chunks_exact_mut(3) {
            px[0] = scene.background[0];
            px[1] = scene.background[1];
            px[2] = scene.background[2];
        }
        rows.push(Mutex::new(row));
    }

    scene.spheres.par_iter().for_each(|sphere| {
        let radius = sphere.radius.round() as i32;
        if radius <= 0 {
            return;
        }

        let consts = precompute_constants(sphere.light_dir, radius, 255);
        let method = resolve_method(scene.shading_method, consts);

        let mut writer = |gx: i32, gy: i32, z: f32, brightness: f32| {
            write_pixel_if_nearer_parallel(gx, gy, z, brightness, sphere, width, height, &rows)
        };

        render_sphere_with_method(sphere, radius, consts, method, &mut writer);
    });

    let mut rgb = vec![0u8; width * height * 3];
    let mut z_buffer = vec![f32::INFINITY; width * height];

    for (y, row_mtx) in rows.iter().enumerate() {
        let row = row_mtx.lock().expect("row lock poisoned");
        let z_dst = &mut z_buffer[y * width..(y + 1) * width];
        let rgb_dst = &mut rgb[y * width * 3..(y + 1) * width * 3];
        z_dst.copy_from_slice(&row.z);
        rgb_dst.copy_from_slice(&row.rgb);
    }

    RenderResult {
        width,
        height,
        rgb,
        z_buffer,
    }
}

fn render_sphere_with_method<F>(
    sphere: &Sphere,
    radius: i32,
    consts: Constants,
    method: ShadingMethod,
    writer: &mut F,
) where
    F: FnMut(i32, i32, f32, f32),
{
    match method {
        ShadingMethod::Auto => unreachable!(),
        ShadingMethod::Direct => render_sphere_direct(sphere, radius, consts, writer),
        ShadingMethod::Diff => render_sphere_diff(sphere, radius, consts, writer),
        ShadingMethod::DiffView => render_sphere_diff_view(sphere, radius, consts, writer),
    }
}

fn resolve_method(method: ShadingMethod, consts: Constants) -> ShadingMethod {
    if method != ShadingMethod::Auto {
        return method;
    }

    if consts.f.abs() <= 1e-6 && consts.g.abs() <= 1e-6 {
        ShadingMethod::DiffView
    } else {
        ShadingMethod::Diff
    }
}

fn precompute_constants(light_dir: [f32; 3], radius: i32, max_index: i32) -> Constants {
    let mut lx = light_dir[0] as f64;
    let mut ly = light_dir[1] as f64;
    let mut lz = light_dir[2] as f64;

    let lnorm = (lx * lx + ly * ly + lz * lz).sqrt();
    if lnorm == 0.0 {
        lx = 0.0;
        ly = 0.0;
        lz = 1.0;
    } else {
        lx /= lnorm;
        ly /= lnorm;
        lz /= lnorm;
    }

    let r = radius as f64;
    let b_c = (lx * lx + ly * ly + lz * lz).sqrt();
    let bp_c = 1.0 / (b_c * r * r);

    let (mantissa, exponent) = frexp_positive(bp_c);
    let rho = (mantissa * (1_i64 << 8) as f64).round() as i64;

    let d = max_index as i64;
    let dp = d * rho;
    let dpp = dp as f64 * r;

    let e = dp as f64 * lz;
    let f = dpp * lx;
    let g = dpp * ly;
    let h = -dpp * lz * r;

    let _c_c = exponent - 8;

    Constants {
        e,
        f,
        g,
        h,
        k: 2.0 * e,
        l: e + f,
    }
}

fn frexp_positive(x: f64) -> (f64, i32) {
    if x <= 0.0 {
        return (0.0, 0);
    }

    let exponent = x.log2().floor() as i32 + 1;
    let mantissa = x / 2f64.powi(exponent);
    (mantissa, exponent)
}

fn circle_half_widths(radius: i32) -> Vec<i32> {
    let mut widths = vec![-1; (radius + 1) as usize];

    let mut x = radius;
    let mut y = 0;
    let mut d = 1 - x;

    while x >= y {
        if y <= radius {
            let i = y as usize;
            widths[i] = widths[i].max(x);
        }
        if x <= radius {
            let i = x as usize;
            widths[i] = widths[i].max(y);
        }

        y += 1;
        if d < 0 {
            d += 2 * y + 1;
        } else {
            x -= 1;
            d += 2 * (y - x) + 1;
        }
    }

    for i in (0..radius as usize).rev() {
        if widths[i] < 0 {
            widths[i] = widths[i + 1];
        }
    }

    widths
}

fn brightness_from_f(f_val: f64, denom: f64, sphere: &Sphere) -> f32 {
    let index = ((f_val.max(0.0) / denom) * 255.0).clamp(0.0, 255.0) as f32;
    let mut bright = sphere.k_d * (index / 255.0) + sphere.k_a;
    bright = bright.clamp(0.0, 1.0);

    let gamma = if sphere.gamma > 0.0 {
        sphere.gamma
    } else {
        1.0
    };
    bright.powf(1.0 / gamma)
}

fn write_pixel_if_nearer_seq(
    gx: i32,
    gy: i32,
    z: f32,
    brightness: f32,
    sphere: &Sphere,
    width: usize,
    height: usize,
    rgb: &mut [u8],
    z_buffer: &mut [f32],
) {
    if gx < 0 || gy < 0 || gx >= width as i32 || gy >= height as i32 {
        return;
    }

    let idx = gy as usize * width + gx as usize;
    if z >= z_buffer[idx] {
        return;
    }

    z_buffer[idx] = z;
    let p = idx * 3;
    rgb[p] = (sphere.base_color[0] as f32 * brightness) as u8;
    rgb[p + 1] = (sphere.base_color[1] as f32 * brightness) as u8;
    rgb[p + 2] = (sphere.base_color[2] as f32 * brightness) as u8;
}

#[cfg(feature = "parallel")]
fn write_pixel_if_nearer_parallel(
    gx: i32,
    gy: i32,
    z: f32,
    brightness: f32,
    sphere: &Sphere,
    width: usize,
    height: usize,
    rows: &[Mutex<RowData>],
) {
    if gx < 0 || gy < 0 || gx >= width as i32 || gy >= height as i32 {
        return;
    }

    let y = gy as usize;
    let x = gx as usize;

    let mut row = rows[y].lock().expect("row lock poisoned");
    if z >= row.z[x] {
        return;
    }

    row.z[x] = z;
    let p = x * 3;
    row.rgb[p] = (sphere.base_color[0] as f32 * brightness) as u8;
    row.rgb[p + 1] = (sphere.base_color[1] as f32 * brightness) as u8;
    row.rgb[p + 2] = (sphere.base_color[2] as f32 * brightness) as u8;
}

fn render_sphere_direct<F>(sphere: &Sphere, radius: i32, consts: Constants, writer: &mut F)
where
    F: FnMut(i32, i32, f32, f32),
{
    let cx = sphere.center[0].round() as i32;
    let cy = sphere.center[1].round() as i32;
    let r2 = radius * radius;

    let mut max_positive_f = 0.0f64;

    for y in -radius..=radius {
        for x in -radius..=radius {
            let d2 = x * x + y * y;
            if d2 > r2 {
                continue;
            }

            let f_val = consts.e * d2 as f64 + consts.f * x as f64 + consts.g * y as f64 + consts.h;
            if f_val > max_positive_f {
                max_positive_f = f_val;
            }
        }
    }

    let denom = if max_positive_f > 0.0 {
        max_positive_f
    } else {
        1.0
    };
    let r = radius as f32;
    let z_bias = sphere.z0 * r - (r * r);

    for y in -radius..=radius {
        for x in -radius..=radius {
            let d2 = x * x + y * y;
            if d2 > r2 {
                continue;
            }

            let f_val = consts.e * d2 as f64 + consts.f * x as f64 + consts.g * y as f64 + consts.h;
            let bright = brightness_from_f(f_val, denom, sphere);
            let z = (d2 as f32 + z_bias) / r;
            writer(cx + x, cy + y, z, bright);
        }
    }
}

fn render_sphere_diff<F>(sphere: &Sphere, radius: i32, consts: Constants, writer: &mut F)
where
    F: FnMut(i32, i32, f32, f32),
{
    let size = (2 * radius + 1) as usize;
    let center = radius;

    let mut f_vals = vec![0.0f64; size * size];
    let mut z_vals = vec![f32::INFINITY; size * size];
    let mut mask = vec![false; size * size];

    let widths = circle_half_widths(radius);
    let r = radius as f32;
    let r2 = radius * radius;
    let z_bias = sphere.z0 * r - (r * r);

    let mut max_positive_f = 0.0f64;

    for y in -radius..=radius {
        let analytic_x_max = ((r2 - y * y) as f64).sqrt().floor() as i32;
        let x_max = widths[y.unsigned_abs() as usize].max(analytic_x_max);
        let x_left = -x_max;

        let mut f_val = consts.e * ((x_left * x_left + y * y) as f64)
            + consts.f * x_left as f64
            + consts.g * y as f64
            + consts.h;
        let mut n_val = 2.0 * consts.e * x_left as f64 + consts.l;

        let mut z_num = (x_left * x_left + y * y) as f32 + z_bias;
        let mut z_step = (2 * x_left + 1) as f32;

        let row = (y + center) as usize;

        for x in x_left..=x_max {
            if x * x + y * y > r2 {
                f_val += n_val;
                n_val += consts.k;
                z_num += z_step;
                z_step += 2.0;
                continue;
            }

            let col = (x + center) as usize;
            let idx = row * size + col;
            mask[idx] = true;
            f_vals[idx] = f_val;
            z_vals[idx] = z_num / r;

            if f_val > max_positive_f {
                max_positive_f = f_val;
            }

            f_val += n_val;
            n_val += consts.k;
            z_num += z_step;
            z_step += 2.0;
        }
    }

    let denom = if max_positive_f > 0.0 {
        max_positive_f
    } else {
        1.0
    };
    let cx = sphere.center[0].round() as i32;
    let cy = sphere.center[1].round() as i32;

    for row in 0..size {
        for col in 0..size {
            let idx = row * size + col;
            if !mask[idx] {
                continue;
            }

            let x = col as i32 - center;
            let y = row as i32 - center;
            let bright = brightness_from_f(f_vals[idx], denom, sphere);
            writer(cx + x, cy + y, z_vals[idx], bright);
        }
    }
}

fn render_sphere_diff_view<F>(sphere: &Sphere, radius: i32, consts: Constants, writer: &mut F)
where
    F: FnMut(i32, i32, f32, f32),
{
    let size = (2 * radius + 1) as usize;
    let center = radius;

    let mut f_vals = vec![0.0f64; size * size];
    let mut z_vals = vec![f32::INFINITY; size * size];
    let mut mask = vec![false; size * size];

    let widths = circle_half_widths(radius);
    let r = radius as f32;
    let r2 = radius * radius;
    let z_bias = sphere.z0 * r - (r * r);

    for y in 0..=radius {
        let analytic_x_max = ((r2 - y * y) as f64).sqrt().floor() as i32;
        let x_max = widths[y as usize].max(analytic_x_max);
        if x_max < y {
            continue;
        }

        let mut x = y;
        let mut d2 = x * x + y * y;
        let mut step = 2 * x + 1;

        while x <= x_max {
            if x * x + y * y > r2 {
                d2 += step;
                step += 2;
                x += 1;
                continue;
            }

            let f_val = consts.e * d2 as f64 + consts.h;
            let z_val = (d2 as f32 + z_bias) / r;

            let mirrored = [
                (x, y),
                (y, x),
                (-x, y),
                (-y, x),
                (x, -y),
                (y, -x),
                (-x, -y),
                (-y, -x),
            ];

            for (mx, my) in mirrored {
                let row = (my + center) as usize;
                let col = (mx + center) as usize;
                let idx = row * size + col;
                mask[idx] = true;
                f_vals[idx] = f_val;
                z_vals[idx] = z_val;
            }

            d2 += step;
            step += 2;
            x += 1;
        }
    }

    let mut max_positive_f = 0.0f64;
    for i in 0..mask.len() {
        if !mask[i] {
            continue;
        }
        let f = f_vals[i].max(0.0);
        if f > max_positive_f {
            max_positive_f = f;
        }
    }

    let denom = if max_positive_f > 0.0 {
        max_positive_f
    } else {
        1.0
    };
    let cx = sphere.center[0].round() as i32;
    let cy = sphere.center[1].round() as i32;

    for row in 0..size {
        for col in 0..size {
            let idx = row * size + col;
            if !mask[idx] {
                continue;
            }

            let x = col as i32 - center;
            let y = row as i32 - center;
            let bright = brightness_from_f(f_vals[idx], denom, sphere);
            writer(cx + x, cy + y, z_vals[idx], bright);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_parses_with_default_method() {
        let data = r#"{
            "width": 32,
            "height": 24,
            "background": [0, 0, 0],
            "spheres": [{
                "center": [12.0, 12.0],
                "radius": 8.0,
                "base_color": [200, 120, 80],
                "light_dir": [0.3, 0.4, 1.0],
                "z0": -4.0,
                "k_d": 1.0,
                "k_a": 0.1,
                "gamma": 2.2
            }]
        }"#;

        let scene = Scene::from_json_str(data).expect("scene parse failed");
        assert_eq!(scene.shading_method, ShadingMethod::Auto);
        assert!(!scene.parallel);
    }

    #[test]
    fn render_produces_non_background_pixels() {
        let scene = Scene {
            width: 64,
            height: 64,
            background: [0, 0, 0],
            shading_method: ShadingMethod::Direct,
            parallel: false,
            spheres: vec![Sphere {
                center: [32.0, 32.0],
                radius: 16.0,
                base_color: [200, 180, 120],
                light_dir: [0.3, 0.4, 1.0],
                z0: -8.0,
                k_d: 1.0,
                k_a: 0.1,
                gamma: 2.2,
            }],
        };

        let out = render_scene(&scene);
        let non_bg = out
            .rgb
            .chunks_exact(3)
            .filter(|p| p[0] != 0 || p[1] != 0 || p[2] != 0)
            .count();
        assert!(non_bg > 0);
    }
}
