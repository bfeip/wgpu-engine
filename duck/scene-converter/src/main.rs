use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};
use image::imageops::FilterType;
use image::GenericImageView;
use duck_engine_import_export::format::{CompressionLevel, SaveOptions, save_to_file_with_options};
use duck_engine_import_export::gltf::load_gltf_scene_from_path;
use duck_engine_scene::Scene;

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

    /// Preprocess environment maps into IBL data (irradiance + prefiltered cubemaps)
    /// so they can be used on platforms without compute shader support
    #[arg(long)]
    bake_ibl: bool,

    /// When baking IBL, also keep the raw HDR source bytes in the output file
    #[arg(long)]
    keep_hdr: bool,
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
        for texture in scene.textures_mut() {
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

    // Bake IBL data if requested
    if cli.bake_ibl && scene.has_environment_maps() {
        eprintln!("Baking IBL data...");
        let renderer = pollster::block_on(
            duck_engine_renderer::Renderer::new_headless(1, 1)
        );

        // Process all environments first (immutable borrow), then attach results (mutable)
        let results: Vec<_> = scene.environment_maps()
            .map(|env_map| {
                eprintln!("  Processing environment map {}...", env_map.id);
                let preprocessed = renderer.preprocess_ibl(env_map)?;
                Ok((env_map.id, preprocessed))
            })
            .collect::<Result<_>>()?;

        for (id, preprocessed) in results {
            let env_map = scene.get_environment_map_mut(id).unwrap();
            env_map.set_preprocessed_ibl(preprocessed);
            if !cli.keep_hdr {
                env_map.drop_source();
            }
        }
        eprintln!("  IBL baking complete.");
    }

    print_stats(&scene);

    eprintln!("Saving to {}...", output.display());
    save_to_file_with_options(&scene, &output, &options)?;

    let file_size = std::fs::metadata(&output)?.len();
    eprintln!("Done. Output: {} ({})", output.display(), format_bytes(file_size));

    Ok(())
}

fn print_stats(scene: &Scene) {
    eprintln!("  Meshes:    {}", scene.mesh_count());
    eprintln!("  Materials: {}", scene.material_count());
    eprintln!("  Textures:  {}", scene.texture_count());
    eprintln!("  Nodes:     {}", scene.node_count());
    eprintln!("  Lights:    {}", scene.lights().len());
    eprintln!("  Instances: {}", scene.instance_count());
    eprintln!("  Env maps:  {}", scene.environment_map_count());
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
