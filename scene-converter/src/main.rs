use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};
use image::imageops::FilterType;
use image::GenericImageView;
use wgpu_engine_scene::format::{CompressionLevel, SaveOptions};
use wgpu_engine_scene::gltf::load_gltf_scene_from_path;
use wgpu_engine_scene::Scene;

const MAX_TEXTURE_DIMENSION: u32 = 2048;

#[derive(Clone, Copy, ValueEnum)]
enum CliCompressionLevel {
    Fast,
    Default,
    Best,
}

impl From<CliCompressionLevel> for CompressionLevel {
    fn from(cli: CliCompressionLevel) -> Self {
        match cli {
            CliCompressionLevel::Fast => CompressionLevel::Fast,
            CliCompressionLevel::Default => CompressionLevel::Default,
            CliCompressionLevel::Best => CompressionLevel::Best,
        }
    }
}

#[derive(Parser)]
#[command(name = "scene-converter")]
#[command(about = "Convert glTF and HDR files to .wgsc scene format")]
#[command(version)]
struct Cli {
    /// Input files (.gltf, .glb, or .hdr)
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Output path (defaults to first input filename with .wgsc extension)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Compression level: fast, default, or best
    #[arg(short, long, default_value = "default")]
    compression: CliCompressionLevel,

    /// Don't resize textures larger than 2048px
    #[arg(long)]
    no_texture_resize: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let options = SaveOptions {
        compression: cli.compression.into(),
        ..Default::default()
    };

    // Classify inputs by extension
    let mut gltf_files = Vec::new();
    let mut hdr_files = Vec::new();

    for path in &cli.inputs {
        match path.extension().and_then(|e| e.to_str()) {
            Some("gltf" | "glb") => gltf_files.push(path),
            Some("hdr") => hdr_files.push(path),
            _ => bail!("Unsupported file type: {}", path.display()),
        }
    }

    if gltf_files.is_empty() {
        bail!("At least one glTF file (.gltf or .glb) is required");
    }
    if gltf_files.len() > 1 {
        bail!("Only one glTF file can be specified");
    }

    let gltf_path = gltf_files[0];
    let output = cli.output.unwrap_or_else(|| gltf_path.with_extension("wgsc"));

    eprintln!("Loading {}...", gltf_path.display());
    let result = load_gltf_scene_from_path(gltf_path, 1.0)?;
    let mut scene = result.scene;

    // Add environment maps from HDR files
    let mut first_env_id = None;
    for hdr_path in &hdr_files {
        eprintln!("Adding HDR: {}", hdr_path.display());
        let id = scene.add_environment_map_from_hdr_path(hdr_path);
        if first_env_id.is_none() {
            first_env_id = Some(id);
        }
    }
    if let Some(id) = first_env_id {
        scene.set_active_environment_map(Some(id));
    }

    // Resize large textures
    if !cli.no_texture_resize {
        for texture in scene.textures.values_mut() {
            let id = texture.id();
            let img = texture.get_image()?;
            let (w, h) = img.dimensions();
            if w > MAX_TEXTURE_DIMENSION || h > MAX_TEXTURE_DIMENSION {
                eprintln!("  Resizing texture {} ({}x{} -> fit {}px)", id, w, h, MAX_TEXTURE_DIMENSION);
                let resized = img.resize(MAX_TEXTURE_DIMENSION, MAX_TEXTURE_DIMENSION, FilterType::Lanczos3);
                texture.set_image(resized);
            }
        }
    }

    print_stats(&scene);

    eprintln!("Saving to {}...", output.display());
    scene.save_to_file_with_options(&output, &options)?;

    let file_size = std::fs::metadata(&output)?.len();
    eprintln!("Done. Output: {} ({})", output.display(), format_bytes(file_size));

    Ok(())
}

fn print_stats(scene: &Scene) {
    eprintln!("  Meshes:    {}", scene.meshes.len());
    eprintln!("  Materials: {}", scene.materials.len());
    eprintln!("  Textures:  {}", scene.textures.len());
    eprintln!("  Nodes:     {}", scene.nodes.len());
    eprintln!("  Lights:    {}", scene.lights.len());
    eprintln!("  Instances: {}", scene.instances.len());
    eprintln!("  Env maps:  {}", scene.environment_maps.len());
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
