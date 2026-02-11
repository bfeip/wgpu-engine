use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use wgpu_engine_scene::format::SaveOptions;
use wgpu_engine_scene::gltf::load_gltf_scene_from_path;
use wgpu_engine_scene::Scene;

#[derive(Parser)]
#[command(name = "scene-converter")]
#[command(about = "Convert glTF files to .wgsc scene format")]
#[command(version)]
struct Cli {
    /// Path to the input glTF file (.gltf or .glb)
    input: PathBuf,

    /// Output path (defaults to input filename with .wgsc extension)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Compression level (1-22, default 3). Lower = faster, higher = smaller.
    #[arg(short, long)]
    compression: Option<i32>,

    /// Use fastest compression (level 1)
    #[arg(long)]
    fast: bool,

    /// Use best compression (level 19)
    #[arg(long)]
    best: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let output = cli.output.unwrap_or_else(|| cli.input.with_extension("wgsc"));

    let options = if cli.fast {
        SaveOptions::fast()
    } else if cli.best {
        SaveOptions::best()
    } else {
        SaveOptions::with_compression_level(cli.compression.unwrap_or(3))
    };

    eprintln!("Loading {}...", cli.input.display());
    let result = load_gltf_scene_from_path(&cli.input, 1.0)?;
    let mut scene = result.scene;

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
