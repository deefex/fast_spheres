use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ShadingMethod {
    Auto,
    Direct,
    Diff,
    DiffView,
}

impl Default for ShadingMethod {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sphere {
    pub center: [f32; 2],
    pub radius: f32,
    pub base_color: [u8; 3],
    pub light_dir: [f32; 3],
    pub z0: f32,
    pub k_d: f32,
    pub k_a: f32,
    pub gamma: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub width: u32,
    pub height: u32,
    pub background: [u8; 3],
    #[serde(default)]
    pub shading_method: ShadingMethod,
    #[serde(default)]
    pub parallel: bool,
    pub spheres: Vec<Sphere>,
}

impl Scene {
    pub fn from_json_str(data: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(data)
    }
}
