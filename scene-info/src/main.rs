//! Scene file analyzer tool.
//!
//! Displays detailed information about .wgsc scene files including:
//! - File structure and section sizes
//! - Compression ratios
//! - Mesh, material, texture, and node statistics
//!
//! Usage: cargo run -p scene-info -- <file.wgsc> [--verbose|-v]

use std::collections::HashMap;
use std::fs;
use std::io::Cursor;
use std::path::Path;

use wgpu_engine::scene::format::{
    FileHeader, FormatError, SectionType, SerializedAnnotation, SerializedInstance,
    SerializedLight, SerializedMaterial, SerializedMesh, SerializedMetadata, SerializedNode,
    SerializedTexture, TableOfContents, TextureFormat, TocEntry,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <file.wgsc> [--verbose|-v]", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");

    if let Err(e) = analyze_scene(path, verbose) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn analyze_scene(path: &str, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(path);
    let bytes = fs::read(path)?;
    let file_size = bytes.len() as u64;

    // Read header
    let mut cursor = Cursor::new(&bytes);
    let header = FileHeader::read(&mut cursor)?;

    // Read TOC
    let toc = read_toc(&bytes, header.toc_offset)?;

    // Print file info
    let file_name = path.file_name().unwrap_or_default().to_string_lossy();
    println!("Scene File: {}", file_name);
    println!("File Size: {}", format_bytes(file_size));
    println!();

    // Print header info
    let version_major = (header.version >> 8) as u8;
    let version_minor = (header.version & 0xFF) as u8;
    println!("Header:");
    println!("  Version: {}.{}", version_major, version_minor);
    println!("  Flags: 0x{:04X}", header.flags);
    println!();

    // Print section breakdown
    print_section_breakdown(&toc, file_size);

    // Read and analyze each section
    let stats = gather_statistics(&bytes, &toc)?;

    // Print scene contents summary
    print_scene_summary(&stats);

    // Print largest textures
    if !stats.textures.is_empty() {
        print_largest_textures(&stats.textures);
    }

    // Verbose output
    if verbose {
        println!();
        print_verbose_details(&stats);
    }

    Ok(())
}

fn read_toc(bytes: &[u8], toc_offset: u64) -> Result<TableOfContents, FormatError> {
    let toc_data = &bytes[toc_offset as usize..];
    let decompressed = zstd::decode_all(toc_data)
        .map_err(|e| FormatError::DecompressionError(e.to_string()))?;
    bincode::deserialize(&decompressed)
        .map_err(|e| FormatError::DeserializationError(e.to_string()))
}

fn read_section<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    entry: &TocEntry,
) -> Result<T, FormatError> {
    let start = entry.offset as usize;
    let end = start + entry.compressed_size as usize;
    let compressed = &bytes[start..end];

    let decompressed = zstd::decode_all(compressed)
        .map_err(|e| FormatError::DecompressionError(e.to_string()))?;

    bincode::deserialize(&decompressed)
        .map_err(|e| FormatError::DeserializationError(e.to_string()))
}

fn print_section_breakdown(toc: &TableOfContents, file_size: u64) {
    println!("Sections:");
    println!(
        "  {:<16} {:>12} {:>14} {:>8} {:>10}",
        "Section", "Compressed", "Uncompressed", "Ratio", "% of File"
    );
    println!("  {}", "-".repeat(64));

    for entry in &toc.entries {
        let section_name = format!("{:?}", entry.section_type);
        let ratio = if entry.compressed_size > 0 {
            entry.uncompressed_size as f64 / entry.compressed_size as f64
        } else {
            1.0
        };
        let percent = (entry.compressed_size as f64 / file_size as f64) * 100.0;

        println!(
            "  {:<16} {:>12} {:>14} {:>7.1}x {:>9.1}%",
            section_name,
            format_bytes(entry.compressed_size),
            format_bytes(entry.uncompressed_size),
            ratio,
            percent
        );
    }
    println!();
}

struct SceneStats {
    metadata: Option<SerializedMetadata>,
    nodes: Vec<SerializedNode>,
    instances: Vec<SerializedInstance>,
    materials: Vec<SerializedMaterial>,
    meshes: Vec<SerializedMesh>,
    textures: Vec<SerializedTexture>,
    lights: Vec<SerializedLight>,
    annotations: Vec<SerializedAnnotation>,
}

fn gather_statistics(bytes: &[u8], toc: &TableOfContents) -> Result<SceneStats, FormatError> {
    let mut stats = SceneStats {
        metadata: None,
        nodes: Vec::new(),
        instances: Vec::new(),
        materials: Vec::new(),
        meshes: Vec::new(),
        textures: Vec::new(),
        lights: Vec::new(),
        annotations: Vec::new(),
    };

    for entry in &toc.entries {
        match entry.section_type {
            SectionType::Metadata => {
                stats.metadata = Some(read_section(bytes, entry)?);
            }
            SectionType::Nodes => {
                stats.nodes = read_section(bytes, entry)?;
            }
            SectionType::Instances => {
                stats.instances = read_section(bytes, entry)?;
            }
            SectionType::Materials => {
                stats.materials = read_section(bytes, entry)?;
            }
            SectionType::Meshes => {
                stats.meshes = read_section(bytes, entry)?;
            }
            SectionType::Textures => {
                stats.textures = read_section(bytes, entry)?;
            }
            SectionType::Lights => {
                stats.lights = read_section(bytes, entry)?;
            }
            SectionType::Annotations => {
                stats.annotations = read_section(bytes, entry)?;
            }
            SectionType::EnvironmentMaps => {
                // Skip for now - environment maps handled separately
            }
        }
    }

    Ok(stats)
}

fn print_scene_summary(stats: &SceneStats) {
    println!("Scene Contents:");

    // Mesh stats
    let total_vertices: usize = stats.meshes.iter().map(|m| m.vertices.len() / 36).sum();
    let total_indices: usize = stats
        .meshes
        .iter()
        .flat_map(|m| m.primitives.iter())
        .map(|p| p.indices.len())
        .sum();
    let total_triangles = total_indices / 3;
    println!(
        "  Meshes: {} ({} vertices, {} triangles)",
        stats.meshes.len(),
        format_number(total_vertices),
        format_number(total_triangles)
    );

    // Material stats
    let materials_with_textures = stats
        .materials
        .iter()
        .filter(|m| {
            m.base_color_texture_id.is_some()
                || m.normal_texture_id.is_some()
                || m.metallic_roughness_texture_id.is_some()
        })
        .count();
    println!(
        "  Materials: {} ({} with textures)",
        stats.materials.len(),
        materials_with_textures
    );

    // Texture stats
    let total_texture_data: u64 = stats.textures.iter().map(|t| t.data.len() as u64).sum();
    println!(
        "  Textures: {} ({})",
        stats.textures.len(),
        format_bytes(total_texture_data)
    );

    // Node stats
    let max_depth = compute_max_depth(&stats.nodes);
    println!("  Nodes: {} (max depth: {})", stats.nodes.len(), max_depth);

    // Lights
    println!("  Lights: {}", stats.lights.len());

    // Annotations
    println!("  Annotations: {}", stats.annotations.len());
    println!();
}

fn compute_max_depth(nodes: &[SerializedNode]) -> usize {
    if nodes.is_empty() {
        return 0;
    }

    // Build parent map
    let parent_map: HashMap<u32, Option<u32>> = nodes.iter().map(|n| (n.id, n.parent_id)).collect();

    let mut max_depth = 0;
    for node in nodes {
        let mut depth = 0;
        let mut current_id = Some(node.id);
        while let Some(id) = current_id {
            depth += 1;
            current_id = parent_map.get(&id).and_then(|&p| p);
        }
        max_depth = max_depth.max(depth);
    }
    max_depth
}

fn print_largest_textures(textures: &[SerializedTexture]) {
    println!("Largest Textures:");
    println!(
        "  {:>3} {:>4} {:>6} {:>12} {:>12}",
        "#", "ID", "Format", "Dimensions", "Data Size"
    );
    println!("  {}", "-".repeat(42));

    let mut sorted: Vec<_> = textures.iter().collect();
    sorted.sort_by(|a, b| b.data.len().cmp(&a.data.len()));

    for (i, tex) in sorted.iter().take(5).enumerate() {
        let format_name = match tex.format {
            TextureFormat::Png => "PNG",
            TextureFormat::Jpeg => "JPEG",
            TextureFormat::Raw => "Raw",
        };
        println!(
            "  {:>3} {:>4} {:>6} {:>5}x{:<5} {:>12}",
            i + 1,
            tex.id,
            format_name,
            tex.width,
            tex.height,
            format_bytes(tex.data.len() as u64)
        );
    }
    println!();
}

fn print_verbose_details(stats: &SceneStats) {
    // Mesh details
    if !stats.meshes.is_empty() {
        println!("Mesh Details:");
        println!(
            "  {:>4} {:>10} {:>10} {:>10} {:>12}",
            "ID", "Vertices", "Triangles", "Lines", "Vertex Data"
        );
        println!("  {}", "-".repeat(50));

        for mesh in &stats.meshes {
            let vertex_count = mesh.vertices.len() / 36;
            let mut tri_count = 0;
            let mut line_count = 0;
            for prim in &mesh.primitives {
                match prim.primitive_type {
                    0 => tri_count += prim.indices.len() / 3,
                    1 => line_count += prim.indices.len() / 2,
                    _ => {}
                }
            }

            println!(
                "  {:>4} {:>10} {:>10} {:>10} {:>12}",
                mesh.id,
                format_number(vertex_count),
                format_number(tri_count),
                format_number(line_count),
                format_bytes(mesh.vertices.len() as u64)
            );
        }
        println!();
    }

    // Material details
    if !stats.materials.is_empty() {
        println!("Material Details:");
        println!(
            "  {:>4} {:>10} {:>10} {:>10} {:>12}",
            "ID", "Base Tex", "Normal", "MetRough", "Base Color"
        );
        println!("  {}", "-".repeat(56));

        for mat in &stats.materials {
            let base_tex = mat
                .base_color_texture_id
                .map(|id| id.to_string())
                .unwrap_or_else(|| "-".to_string());
            let normal_tex = mat
                .normal_texture_id
                .map(|id| id.to_string())
                .unwrap_or_else(|| "-".to_string());
            let mr_tex = mat
                .metallic_roughness_texture_id
                .map(|id| id.to_string())
                .unwrap_or_else(|| "-".to_string());
            let base_color = format!(
                "#{:02X}{:02X}{:02X}",
                (mat.base_color_factor[0] * 255.0) as u8,
                (mat.base_color_factor[1] * 255.0) as u8,
                (mat.base_color_factor[2] * 255.0) as u8,
            );

            println!(
                "  {:>4} {:>10} {:>10} {:>10} {:>12}",
                mat.id, base_tex, normal_tex, mr_tex, base_color
            );
        }
        println!();
    }

    // All textures detail
    if !stats.textures.is_empty() {
        println!("Texture Details:");
        println!(
            "  {:>4} {:>6} {:>12} {:>12}",
            "ID", "Format", "Dimensions", "Data Size"
        );
        println!("  {}", "-".repeat(40));

        for tex in &stats.textures {
            let format_name = match tex.format {
                TextureFormat::Png => "PNG",
                TextureFormat::Jpeg => "JPEG",
                TextureFormat::Raw => "Raw",
            };
            println!(
                "  {:>4} {:>6} {:>5}x{:<5} {:>12}",
                tex.id,
                format_name,
                tex.width,
                tex.height,
                format_bytes(tex.data.len() as u64)
            );
        }
        println!();
    }

    // Node hierarchy
    if !stats.nodes.is_empty() {
        println!("Node Hierarchy:");
        print_node_tree(&stats.nodes);
        println!();
    }
}

fn print_node_tree(nodes: &[SerializedNode]) {
    // Find root nodes (no parent)
    let roots: Vec<_> = nodes.iter().filter(|n| n.parent_id.is_none()).collect();

    // Build children map
    let mut children_map: HashMap<u32, Vec<&SerializedNode>> = HashMap::new();
    for node in nodes {
        if let Some(parent_id) = node.parent_id {
            children_map.entry(parent_id).or_default().push(node);
        }
    }

    fn print_node(
        node: &SerializedNode,
        children_map: &HashMap<u32, Vec<&SerializedNode>>,
        prefix: &str,
        is_last: bool,
    ) {
        let connector = if is_last { "`-- " } else { "|-- " };
        let name = node.name.as_deref().unwrap_or("<unnamed>");
        let instance_info = node
            .instance_id
            .map(|id| format!(" (instance:{})", id))
            .unwrap_or_default();
        let visibility = if node.visible { "" } else { " [hidden]" };

        println!(
            "  {}{}[{}] {}{}{}",
            prefix, connector, node.id, name, instance_info, visibility
        );

        if let Some(children) = children_map.get(&node.id) {
            let child_prefix = format!("{}{}", prefix, if is_last { "    " } else { "|   " });
            for (i, child) in children.iter().enumerate() {
                let is_last_child = i == children.len() - 1;
                print_node(child, children_map, &child_prefix, is_last_child);
            }
        }
    }

    for (i, root) in roots.iter().enumerate() {
        let is_last = i == roots.len() - 1;
        print_node(root, &children_map, "", is_last);
    }
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

fn format_number(n: usize) -> String {
    if n < 1000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}K", n as f64 / 1000.0)
    } else {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    }
}
