//! Scene file analyzer tool.
//!
//! Displays detailed information about .duck scene files including:
//! - File structure and resource sizes
//! - Compression ratios
//! - Mesh, material, texture, and node statistics

use std::collections::HashMap;
use std::fs;
use std::io::Cursor;
use std::path::PathBuf;

use clap::Parser;
use duck_engine_import_export::format::{
    FileHeader, FormatError, ResourceType, SerializedMetadata,
    TocEntry, decode_resource,
};
use duck_engine_scene::{FaceMaterial, Instance, Mesh, Node, NodeId, NodePayload, Texture, TextureFormat};

#[derive(Parser)]
#[command(name = "scene-info")]
#[command(about = "Analyze .duck scene files and display detailed statistics")]
#[command(version)]
struct Cli {
    /// Path to the .duck scene file
    file: PathBuf,

    /// Show detailed mesh information (vertices, triangles, indices per mesh)
    #[arg(short, long)]
    meshes: bool,

    /// Show detailed texture information (format, dimensions, size per texture)
    #[arg(short, long)]
    textures: bool,

    /// Show detailed material information (texture references, colors)
    #[arg(short = 'M', long)]
    materials: bool,

    /// Show node hierarchy tree
    #[arg(short, long)]
    nodes: bool,

    /// Show all detailed information (equivalent to -m -t -M -n)
    #[arg(short, long)]
    all: bool,

    /// Hide the resource size breakdown table
    #[arg(long)]
    no_sections: bool,

    /// Show the scene contents summary (reads all resources)
    #[arg(short, long)]
    summary: bool,
}

impl Cli {
    fn show_meshes(&self) -> bool {
        self.meshes || self.all
    }

    fn show_textures(&self) -> bool {
        self.textures || self.all
    }

    fn show_materials(&self) -> bool {
        self.materials || self.all
    }

    fn show_nodes(&self) -> bool {
        self.nodes || self.all
    }
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = analyze_scene(&cli) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn analyze_scene(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = fs::read(&cli.file)?;
    let file_size = bytes.len() as u64;

    // Read header
    let mut cursor = Cursor::new(&bytes);
    let header = FileHeader::read(&mut cursor)?;

    // Read TOC
    let toc = read_toc(&bytes, header.toc_offset)?;

    // Print file info
    let file_name = cli.file.file_name().unwrap_or_default().to_string_lossy();
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

    // Print resource breakdown
    if !cli.no_sections {
        print_resource_breakdown(&toc, file_size);
    }

    let need_content = cli.summary
        || cli.show_nodes()
        || cli.show_materials()
        || cli.show_meshes()
        || cli.show_textures();

    let stats = if need_content {
        Some(gather_statistics(&bytes, &toc)?)
    } else {
        None
    };

    if let Some(ref stats) = stats {
        if cli.summary {
            print_scene_summary(stats);
        }

        if cli.show_textures() && !stats.textures.is_empty() {
            print_largest_textures(&stats.textures);
        }

        if cli.show_meshes() && !stats.meshes.is_empty() {
            print_largest_meshes(&stats.meshes);
        }

        let show_any_details =
            cli.show_meshes() || cli.show_textures() || cli.show_materials() || cli.show_nodes();

        if show_any_details {
            println!();
        }

        if cli.show_meshes() && !stats.meshes.is_empty() {
            print_mesh_details(&stats.meshes);
        }

        if cli.show_materials() && !stats.materials.is_empty() {
            print_material_details(&stats.materials);
        }

        if cli.show_textures() && !stats.textures.is_empty() {
            print_texture_details(&stats.textures);
        }

        if cli.show_nodes() && !stats.nodes.is_empty() {
            print_node_hierarchy(&stats.nodes);
        }
    }

    Ok(())
}

fn read_toc(bytes: &[u8], toc_offset: u64) -> Result<Vec<TocEntry>, FormatError> {
    let toc_data = &bytes[toc_offset as usize..];
    decode_resource(toc_data)
}

fn print_resource_breakdown(toc: &[TocEntry], file_size: u64) {
    // Group entries by resource type, summing sizes and counting entries
    let mut by_type: HashMap<ResourceType, (u32, u64)> = HashMap::new();
    for entry in toc {
        let (count, total) = by_type.entry(entry.resource_type).or_default();
        *count += 1;
        *total += entry.size as u64;
    }

    println!("Resources:");
    println!(
        "  {:<16} {:>8} {:>12} {:>10}",
        "Type", "Count", "Total Size", "% of File"
    );
    println!("  {}", "-".repeat(50));

    // Fixed display order
    let order = [
        ResourceType::Metadata,
        ResourceType::Node,
        ResourceType::Instance,
        ResourceType::FaceMaterial,
        ResourceType::LineMaterial,
        ResourceType::PointMaterial,
        ResourceType::Mesh,
        ResourceType::Texture,
        ResourceType::EnvironmentMap,
    ];

    for rt in &order {
        if let Some(&(count, total)) = by_type.get(rt) {
            let percent = (total as f64 / file_size as f64) * 100.0;
            println!(
                "  {:<16} {:>8} {:>12} {:>9.1}%",
                format!("{:?}", rt),
                count,
                format_bytes(total),
                percent,
            );
        }
    }
    println!();
}

struct SceneStats {
    metadata: Option<SerializedMetadata>,
    nodes: Vec<Node>,
    instances: Vec<Instance>,
    materials: Vec<FaceMaterial>,
    meshes: Vec<Mesh>,
    textures: Vec<Texture>,
}

fn gather_statistics(bytes: &[u8], toc: &[TocEntry]) -> Result<SceneStats, FormatError> {
    let mut stats = SceneStats {
        metadata: None,
        nodes: Vec::new(),
        instances: Vec::new(),
        materials: Vec::new(),
        meshes: Vec::new(),
        textures: Vec::new(),
    };

    for entry in toc {
        let resource_bytes = {
            let start = entry.offset as usize;
            let end = start + entry.size as usize;
            &bytes[start..end]
        };

        match entry.resource_type {
            ResourceType::Metadata => {
                stats.metadata = Some(decode_resource(resource_bytes)?);
            }
            ResourceType::Node => {
                stats.nodes.push(decode_resource(resource_bytes)?);
            }
            ResourceType::Instance => {
                stats.instances.push(decode_resource(resource_bytes)?);
            }
            ResourceType::FaceMaterial => {
                stats.materials.push(decode_resource(resource_bytes)?);
            }
            ResourceType::LineMaterial | ResourceType::PointMaterial => {
                // Detail view is face-centric; line/point materials are counted
                // via the resource-type table but not detailed here.
            }
            ResourceType::Mesh => {
                stats.meshes.push(decode_resource(resource_bytes)?);
            }
            ResourceType::Texture => {
                let raw = resource_bytes.to_vec();
                let tex = Texture::from_image_bytes_with_id(entry.resource_id.cast(), raw)
                    .map_err(|e| FormatError::TextureError(e.to_string()))?;
                stats.textures.push(tex);
            }
            ResourceType::EnvironmentMap => {
                // Not displayed in current stats views
            }
        }
    }

    Ok(stats)
}

fn print_scene_summary(stats: &SceneStats) {
    println!("Scene Contents:");

    // Mesh stats
    let total_vertices: usize = stats.meshes.iter().map(|m| m.vertices().len()).sum();
    let total_indices: usize = stats
        .meshes
        .iter()
        .flat_map(|m| m.primitives().iter())
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
            m.base_color_texture().is_some()
                || m.normal_texture().is_some()
                || m.metallic_roughness_texture().is_some()
        })
        .count();
    println!(
        "  Materials: {} ({} with textures)",
        stats.materials.len(),
        materials_with_textures
    );

    // Texture stats — sum original_bytes lengths
    let total_texture_data: u64 = stats
        .textures
        .iter()
        .map(|t| t.original_bytes().map_or(0, |(b, _)| b.len()) as u64)
        .sum();
    println!(
        "  Textures: {} ({})",
        stats.textures.len(),
        format_bytes(total_texture_data)
    );

    // Node stats
    let max_depth = compute_max_depth(&stats.nodes);
    println!("  Nodes: {} (max depth: {})", stats.nodes.len(), max_depth);

    println!();
}

fn compute_max_depth(nodes: &[Node]) -> usize {
    if nodes.is_empty() {
        return 0;
    }

    let parent_map: HashMap<NodeId, Option<NodeId>> =
        nodes.iter().map(|n| (n.id, n.parent())).collect();

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

fn print_largest_textures(textures: &[Texture]) {
    println!("Largest Textures:");
    println!(
        "  {:>3} {:>4} {:>6} {:>12} {:>12}",
        "#", "ID", "Format", "Dimensions", "Data Size"
    );
    println!("  {}", "-".repeat(42));

    let mut sorted: Vec<_> = textures.iter().collect();
    sorted.sort_by(|a, b| {
        let a_size = a.original_bytes().map_or(0, |(b, _)| b.len());
        let b_size = b.original_bytes().map_or(0, |(b, _)| b.len());
        b_size.cmp(&a_size)
    });

    for (i, tex) in sorted.iter().take(5).enumerate() {
        let (format_name, data_size) = match tex.original_bytes() {
            Some((b, TextureFormat::Png)) => ("PNG", b.len()),
            Some((b, TextureFormat::Jpeg)) => ("JPEG", b.len()),
            Some((b, TextureFormat::Raw)) => ("Raw", b.len()),
            None => ("?", 0),
        };
        let (w, h) = tex.dimensions().unwrap_or((0, 0));
        println!(
            "  {:>3} {:>4} {:>6} {:>5}x{:<5} {:>12}",
            i + 1,
            tex.id,
            format_name,
            w,
            h,
            format_bytes(data_size as u64)
        );
    }
    println!();
}

fn print_largest_meshes(meshes: &[Mesh]) {
    println!("Largest Meshes:");
    println!(
        "  {:>4} {:>10} {:>10} {:>10}",
        "ID", "Vertices", "Triangles", "Lines"
    );
    println!("  {}", "-".repeat(38));

    use duck_engine_scene::PrimitiveType;

    let mut sorted: Vec<_> = meshes.iter().collect();
    sorted.sort_by(|a, b| b.vertices().len().cmp(&a.vertices().len()));

    for mesh in sorted.iter().take(5) {
        let vertex_count = mesh.vertices().len();
        let mut tri_count = 0;
        let mut line_count = 0;
        for prim in mesh.primitives() {
            match prim.primitive_type {
                PrimitiveType::TriangleList => tri_count += prim.indices.len() / 3,
                PrimitiveType::LineList => line_count += prim.indices.len() / 2,
                PrimitiveType::PointList => {}
            }
        }

        println!(
            "  {:>4} {:>10} {:>10} {:>10}",
            mesh.id,
            format_number(vertex_count),
            format_number(tri_count),
            format_number(line_count),
        );
    }
    println!();
}

fn print_mesh_details(meshes: &[Mesh]) {
    println!("Mesh Details:");
    println!(
        "  {:>4} {:>10} {:>10} {:>10} {:>12}",
        "ID", "Vertices", "Triangles", "Lines", "Vertex Data"
    );
    println!("  {}", "-".repeat(50));

    use duck_engine_scene::PrimitiveType;

    for mesh in meshes {
        let vertex_count = mesh.vertices().len();
        let mut tri_count = 0;
        let mut line_count = 0;
        for prim in mesh.primitives() {
            match prim.primitive_type {
                PrimitiveType::TriangleList => tri_count += prim.indices.len() / 3,
                PrimitiveType::LineList => line_count += prim.indices.len() / 2,
                PrimitiveType::PointList => {}
            }
        }

        let vertex_data_size = std::mem::size_of_val(mesh.vertices());
        println!(
            "  {:>4} {:>10} {:>10} {:>10} {:>12}",
            mesh.id,
            format_number(vertex_count),
            format_number(tri_count),
            format_number(line_count),
            format_bytes(vertex_data_size as u64)
        );
    }
    println!();
}

fn print_material_details(materials: &[FaceMaterial]) {
    println!("Material Details:");
    println!(
        "  {:>4} {:>10} {:>10} {:>10} {:>12}",
        "ID", "Base Tex", "Normal", "MetRough", "Base Color"
    );
    println!("  {}", "-".repeat(56));

    for mat in materials {
        let base_tex = mat
            .base_color_texture()
            .map(|id| id.to_string())
            .unwrap_or_else(|| "-".to_string());
        let normal_tex = mat
            .normal_texture()
            .map(|id| id.to_string())
            .unwrap_or_else(|| "-".to_string());
        let mr_tex = mat
            .metallic_roughness_texture()
            .map(|id| id.to_string())
            .unwrap_or_else(|| "-".to_string());
        let base_color = {
            let c = mat.base_color_factor();
            format!(
                "#{:02X}{:02X}{:02X}",
                (c.r * 255.0) as u8,
                (c.g * 255.0) as u8,
                (c.b * 255.0) as u8,
            )
        };

        println!(
            "  {:>4} {:>10} {:>10} {:>10} {:>12}",
            mat.id, base_tex, normal_tex, mr_tex, base_color
        );
    }
    println!();
}

fn print_texture_details(textures: &[Texture]) {
    println!("Texture Details:");
    println!(
        "  {:>4} {:>6} {:>12} {:>12}",
        "ID", "Format", "Dimensions", "Data Size"
    );
    println!("  {}", "-".repeat(40));

    for tex in textures {
        let (format_name, data_size) = match tex.original_bytes() {
            Some((b, TextureFormat::Png)) => ("PNG", b.len()),
            Some((b, TextureFormat::Jpeg)) => ("JPEG", b.len()),
            Some((b, TextureFormat::Raw)) => ("Raw", b.len()),
            None => ("?", 0),
        };
        let (w, h) = tex.dimensions().unwrap_or((0, 0));
        println!(
            "  {:>4} {:>6} {:>5}x{:<5} {:>12}",
            tex.id,
            format_name,
            w,
            h,
            format_bytes(data_size as u64)
        );
    }
    println!();
}

fn print_node_hierarchy(nodes: &[Node]) {
    println!("Node Hierarchy:");

    let roots: Vec<_> = nodes.iter().filter(|n| n.parent().is_none()).collect();

    let mut children_map: HashMap<NodeId, Vec<&Node>> = HashMap::new();
    for node in nodes {
        if let Some(parent_id) = node.parent() {
            children_map.entry(parent_id).or_default().push(node);
        }
    }

    fn print_node(
        node: &Node,
        children_map: &HashMap<NodeId, Vec<&Node>>,
        prefix: &str,
        is_last: bool,
    ) {
        let connector = if is_last { "`-- " } else { "|-- " };
        let name = node.name.as_deref().unwrap_or("<unnamed>");
        let instance_info = match node.payload() {
            NodePayload::Instance(id) => format!(" (instance:{})", id),
            _ => String::new(),
        };
        let visibility = if node.visibility() == duck_engine_scene::Visibility::Visible { "" } else { " [hidden]" };

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
    println!();
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
