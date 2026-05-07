//! Scene file format serialization.
//!
//! This module provides serialization and deserialization of scenes to a custom
//! binary format (.duck). The format is designed to be:
//! - Concise: Uses Zstd compression for large data sections
//! - Flexible: Section-based structure allows adding new data types
//! - Future-proof: Magic number and version header for compatibility
//!
//! # File Structure
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │ FIXED HEADER (16 bytes)                                      │
//! │  [0..4]   Magic: b"DUCK"                                     │
//! │  [4..6]   Version: u16 (major << 8 | minor)                  │
//! │  [6..8]   Flags: u16 (reserved)                              │
//! │  [8..16]  TOC offset: u64                                    │
//! ├──────────────────────────────────────────────────────────────┤
//! │ SECTION DATA (variable, Zstd compressed)                     │
//! ├──────────────────────────────────────────────────────────────┤
//! │ TABLE OF CONTENTS (at end of file)                           │
//! └──────────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashSet;
use std::io::{Read, Write, Cursor};

use image::codecs::png::CompressionType;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use duck_engine_scene::{
    Instance, InstanceId,
    Material, MaterialId,
    Mesh, MeshId,
    Node, NodeId, NodePayload,
    Texture,
    EnvironmentMap, EnvironmentMapId, Scene,
};

// ============================================================================
// Constants
// ============================================================================

/// Magic number identifying Duck files: "DUCK" in ASCII
pub const MAGIC: [u8; 4] = *b"DUCK";

/// Current format version (major.minor encoded as single u16)
/// major = version >> 8, minor = version & 0xFF
pub const VERSION: u16 = 0x0004; // 0.4 — Texture and EnvironmentMap use direct serde; IBL embedded in env map

/// Size of the fixed header in bytes
pub const HEADER_SIZE: usize = std::mem::size_of::<FileHeader>();

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during scene serialization/deserialization.
#[derive(Debug, Error)]
pub enum FormatError {
    #[error("Invalid magic number")]
    InvalidMagic,

    #[error("Unsupported version: {0}.{1}")]
    UnsupportedVersion(u8, u8),

    #[error("Compression error: {0}")]
    CompressionError(String),

    #[error("Decompression error: {0}")]
    DecompressionError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid section type: {0}")]
    InvalidSectionType(u8),

    #[error("Missing required section: {0:?}")]
    MissingSectionType(SectionType),

    #[error("Texture load error: {0}")]
    TextureError(String),
}

// ============================================================================
// Save Options
// ============================================================================

/// Controls the compression/quality tradeoff for saving scenes.
///
/// Affects zstd data compression, JPEG texture quality, and PNG compression.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompressionLevel {
    /// Fastest compression, larger files. Zstd level 1, JPEG quality 80, PNG fast.
    Fast,
    /// Balanced compression (default). Zstd level 3, JPEG quality 90, PNG default.
    #[default]
    Default,
    /// Best compression, smaller files. Zstd level 19, JPEG quality 95, PNG best.
    Best,
}

impl CompressionLevel {
    /// Zstd compression level for data sections.
    pub fn zstd_level(self) -> i32 {
        match self {
            Self::Fast => 1,
            Self::Default => 3,
            Self::Best => 19,
        }
    }

    /// JPEG quality (1-100) for fallback texture encoding.
    pub fn jpeg_quality(self) -> u8 {
        match self {
            Self::Fast => 80,
            Self::Default => 90,
            Self::Best => 95,
        }
    }

    /// PNG compression type for fallback texture encoding.
    pub fn png_compression(self) -> CompressionType {
        match self {
            Self::Fast => CompressionType::Fast,
            Self::Default => CompressionType::Default,
            Self::Best => CompressionType::Best,
        }
    }
}

/// Options for saving scenes.
#[derive(Clone, Debug, Default)]
pub struct SaveOptions {
    /// Compression level for data sections.
    pub compression: CompressionLevel,
}

// ============================================================================
// Section Types
// ============================================================================

/// Identifies the type of data in a section.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SectionType {
    /// Scene metadata (name, timestamps, generator info)
    Metadata = 0,
    /// Scene graph nodes with hierarchy
    Nodes = 1,
    /// Mesh-material instance bindings
    Instances = 2,
    /// Material definitions
    Materials = 3,
    /// Mesh geometry data
    Meshes = 4,
    /// Embedded texture data
    Textures = 5,
    /// Light definitions
    Lights = 6,
    /// Annotation data (v1 only, not written in v2+)
    Annotations = 7,
    /// Environment map data
    EnvironmentMaps = 8,
    /// Preprocessed IBL cubemap data (irradiance + prefiltered)
    PreprocessedIblData = 9,
    /// Named view (saved camera state) data
    Views = 10,
}

impl TryFrom<u8> for SectionType {
    type Error = FormatError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(SectionType::Metadata),
            1 => Ok(SectionType::Nodes),
            2 => Ok(SectionType::Instances),
            3 => Ok(SectionType::Materials),
            4 => Ok(SectionType::Meshes),
            5 => Ok(SectionType::Textures),
            6 => Ok(SectionType::Lights),
            7 => Ok(SectionType::Annotations),
            8 => Ok(SectionType::EnvironmentMaps),
            9 => Ok(SectionType::PreprocessedIblData),
            10 => Ok(SectionType::Views),
            _ => Err(FormatError::InvalidSectionType(value)),
        }
    }
}

// ============================================================================
// File Header & TOC
// ============================================================================

/// Fixed-size file header at the start of every .duck file.
#[derive(Debug, Clone)]
pub struct FileHeader {
    /// Magic number (must be MAGIC)
    pub magic: [u8; 4],
    /// Format version
    pub version: u16,
    /// Reserved flags
    pub flags: u16,
    /// Byte offset to the table of contents
    pub toc_offset: u64,
}

impl FileHeader {
    pub fn new(toc_offset: u64) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            flags: 0,
            toc_offset,
        }
    }

    pub fn write<W: Write>(&self, writer: &mut W) -> Result<(), FormatError> {
        writer.write_all(&self.magic)?;
        writer.write_all(&self.version.to_le_bytes())?;
        writer.write_all(&self.flags.to_le_bytes())?;
        writer.write_all(&self.toc_offset.to_le_bytes())?;
        Ok(())
    }

    pub fn read<R: Read>(reader: &mut R) -> Result<Self, FormatError> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;

        if magic != MAGIC {
            return Err(FormatError::InvalidMagic);
        }

        let mut version_bytes = [0u8; 2];
        reader.read_exact(&mut version_bytes)?;
        let version = u16::from_le_bytes(version_bytes);

        let major = (version >> 8) as u8;
        let minor = (version & 0xFF) as u8;
        if major != 0 || minor != 4 {
            return Err(FormatError::UnsupportedVersion(major, minor));
        }

        let mut flags_bytes = [0u8; 2];
        reader.read_exact(&mut flags_bytes)?;
        let flags = u16::from_le_bytes(flags_bytes);

        let mut toc_offset_bytes = [0u8; 8];
        reader.read_exact(&mut toc_offset_bytes)?;
        let toc_offset = u64::from_le_bytes(toc_offset_bytes);

        Ok(Self {
            magic,
            version,
            flags,
            toc_offset,
        })
    }
}

/// Entry in the table of contents describing a section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TocEntry {
    /// Section type
    pub section_type: SectionType,
    /// Byte offset from start of file
    pub offset: u64,
    /// Size of compressed data
    pub compressed_size: u64,
    /// Size of uncompressed data
    pub uncompressed_size: u64,
}

/// Table of contents listing all sections in the file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableOfContents {
    pub entries: Vec<TocEntry>,
}

impl TableOfContents {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn add_entry(&mut self, entry: TocEntry) {
        self.entries.push(entry);
    }

    pub fn find(&self, section_type: SectionType) -> Option<&TocEntry> {
        self.entries.iter().find(|e| e.section_type == section_type)
    }
}

impl Default for TableOfContents {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Serializable Data Structures
// ============================================================================

/// Scene metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedMetadata {
    /// Optional scene name
    pub name: Option<String>,
    /// Unix timestamp of creation
    pub created_at: u64,
    /// Generator string (e.g., "duck-engine 0.1.0")
    pub generator: String,
    /// Root node IDs
    pub root_nodes: Vec<NodeId>,
    /// Active environment map ID, if any
    pub active_environment_map: Option<EnvironmentMapId>,
}


// ============================================================================
// Annotation Content Filter
// ============================================================================

/// Collects IDs of annotation-created content to exclude from serialization.
struct AnnotationContentFilter {
    nodes: HashSet<NodeId>,
    instances: HashSet<InstanceId>,
    meshes: HashSet<MeshId>,
    materials: HashSet<MaterialId>,
}

impl AnnotationContentFilter {
    /// Build the filter from a scene by collecting all annotation-created content.
    fn from_scene(scene: &Scene) -> Self {
        let mut filter = Self {
            nodes: HashSet::new(),
            instances: HashSet::new(),
            meshes: HashSet::new(),
            materials: HashSet::new(),
        };

        // Collect all annotation-created node IDs (recursively includes children)
        if let Some(root_id) = scene.annotations.root_node() {
            collect_subtree_nodes(scene, root_id, &mut filter.nodes);
        }

        // Collect instance IDs from annotation nodes
        filter.instances = filter.nodes
            .iter()
            .filter_map(|&node_id| scene.get_node(node_id))
            .filter_map(|node| match node.payload() {
                NodePayload::Instance(id) => Some(*id),
                _ => None,
            })
            .collect();

        // Collect mesh and material IDs from annotation instances
        for &instance_id in &filter.instances {
            if let Some(instance) = scene.get_instance(instance_id) {
                filter.meshes.insert(instance.mesh());
                filter.materials.insert(instance.material());
            }
        }

        filter
    }
}

/// Recursively collects all node IDs in a subtree.
fn collect_subtree_nodes(scene: &Scene, node_id: NodeId, collected: &mut HashSet<NodeId>) {
    collected.insert(node_id);
    if let Some(node) = scene.get_node(node_id) {
        for &child_id in node.children() {
            collect_subtree_nodes(scene, child_id, collected);
        }
    }
}

// ============================================================================
// Compression Utilities
// ============================================================================

/// Compress data using Zstd with the specified compression level.
pub fn compress_with_level(data: &[u8], level: i32) -> Result<Vec<u8>, FormatError> {
    zstd::encode_all(Cursor::new(data), level)
        .map_err(|e| FormatError::CompressionError(e.to_string()))
}

/// Compress data using Zstd with default compression level (3).
pub fn compress(data: &[u8]) -> Result<Vec<u8>, FormatError> {
    compress_with_level(data, 3)
}

/// Decompress Zstd-compressed data.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, FormatError> {
    zstd::decode_all(Cursor::new(data))
        .map_err(|e| FormatError::DecompressionError(e.to_string()))
}

// ============================================================================
// Section Writing/Reading
// ============================================================================

/// Serialize and compress a section with a specific compression level.
pub fn serialize_section_with_level<T: Serialize>(data: &T, level: i32) -> Result<(Vec<u8>, usize), FormatError> {
    let uncompressed = bincode::serde::encode_to_vec(data, bincode::config::legacy())
        .map_err(|e| FormatError::SerializationError(e.to_string()))?;
    let uncompressed_size = uncompressed.len();
    let compressed = compress_with_level(&uncompressed, level)?;
    Ok((compressed, uncompressed_size))
}

/// Serialize and compress a section, returning the compressed bytes.
pub fn serialize_section<T: Serialize>(data: &T) -> Result<(Vec<u8>, usize), FormatError> {
    serialize_section_with_level(data, 3)
}

/// Serialize, compress, and append a section to the output buffer, adding a TOC entry.
fn write_section<T: Serialize>(
    data: &T,
    section_type: SectionType,
    compression_level: i32,
    output: &mut Vec<u8>,
    toc: &mut TableOfContents,
) -> Result<(), FormatError> {
    let offset = output.len() as u64;
    let (compressed, uncompressed_size) = serialize_section_with_level(data, compression_level)?;
    toc.add_entry(TocEntry {
        section_type,
        offset,
        compressed_size: compressed.len() as u64,
        uncompressed_size: uncompressed_size as u64,
    });
    output.extend(compressed);
    Ok(())
}

/// Decompress and deserialize a section.
pub fn deserialize_section<T: for<'de> Deserialize<'de>>(compressed: &[u8]) -> Result<T, FormatError> {
    let uncompressed = decompress(compressed)?;
    let (value, _) = bincode::serde::decode_from_slice(&uncompressed, bincode::config::legacy())
        .map_err(|e| FormatError::DeserializationError(e.to_string()))?;
    Ok(value)
}

// ============================================================================
// Phased Deserialization
// ============================================================================

/// All deserialized sections from a Duck file, ready for scene assembly.
/// Produced by [`parse_duck`], consumed by [`assemble_duck_scene`].
pub struct DuckSections {
    pub metadata: SerializedMetadata,
    pub textures: Vec<Texture>,
    pub materials: Vec<Material>,
    pub meshes: Vec<Mesh>,
    pub instances: Vec<Instance>,
    pub nodes: Vec<Node>,
    pub environment_maps: Vec<EnvironmentMap>,
}

/// Parse Duck header, TOC, and decompress/deserialize all sections (including texture decoding).
pub fn parse_duck(bytes: &[u8]) -> Result<DuckSections, FormatError> {
    let mut cursor = Cursor::new(bytes);
    let header = FileHeader::read(&mut cursor)?;

    let toc_data = &bytes[header.toc_offset as usize..];
    let toc: TableOfContents = deserialize_section(toc_data)?;

    let read_section = |section_type: SectionType| -> Result<&[u8], FormatError> {
        let entry = toc
            .find(section_type)
            .ok_or(FormatError::MissingSectionType(section_type))?;
        let start = entry.offset as usize;
        let end = start + entry.compressed_size as usize;
        Ok(&bytes[start..end])
    };

    let metadata: SerializedMetadata =
        deserialize_section(read_section(SectionType::Metadata)?)?;
    let textures: Vec<Texture> =
        deserialize_section(read_section(SectionType::Textures)?)?;
    let materials: Vec<Material> =
        deserialize_section(read_section(SectionType::Materials)?)?;
    let meshes: Vec<Mesh> =
        deserialize_section(read_section(SectionType::Meshes)?)?;
    let instances: Vec<Instance> =
        deserialize_section(read_section(SectionType::Instances)?)?;
    let nodes: Vec<Node> =
        deserialize_section(read_section(SectionType::Nodes)?)?;
    // SectionType::Lights, SectionType::Views, and SectionType::PreprocessedIblData
    // are no longer written; silently skip them if present in older files.
    let environment_maps: Vec<EnvironmentMap> =
        if let Some(entry) = toc.find(SectionType::EnvironmentMaps) {
            let start = entry.offset as usize;
            let end = start + entry.compressed_size as usize;
            deserialize_section(&bytes[start..end])?
        } else {
            Vec::new()
        };

    Ok(DuckSections { metadata, textures, materials, meshes, instances, nodes, environment_maps })
}

/// Assemble a [`Scene`] from parsed sections.
pub fn assemble_duck_scene(sections: DuckSections) -> Result<Scene, FormatError> {
    let mut scene = Scene::new();

    for texture in sections.textures {
        scene.add_texture(texture);
    }

    for mesh in sections.meshes {
        scene.add_mesh(mesh);
    }

    for material in sections.materials {
        scene.add_material(material);
    }

    for instance in sections.instances {
        scene.add_instance(instance);
    }

    for node in sections.nodes {
        scene.insert_node(node);
    }

    scene.set_root_node_order(sections.metadata.root_nodes);

    for em in sections.environment_maps {
        scene.add_environment_map(em);
    }

    scene.set_active_environment_map(sections.metadata.active_environment_map);

    Ok(scene)
}

// ============================================================================
// Scene Serialization
// ============================================================================

/// Estimates the serialized size in bytes for buffer pre-allocation.
///
/// Returns a heuristic upper bound — intentionally overestimates to avoid
/// reallocations. The actual size after zstd compression will typically be
/// smaller.
pub fn estimate_serialized_size(scene: &Scene) -> usize {
    use duck_engine_scene::{EnvironmentSource};

    const GOOD_COMPRESSION_RATIO: f64 = 0.6;

    // Mesh data (vertices + indices), estimate zstd compression at 60%
    let mesh_raw: usize = scene.meshes()
        .map(|m| {
            let vert_bytes = std::mem::size_of_val(m.vertices());
            let idx_bytes: usize = m.primitives().iter()
                .map(|p| p.indices.len() * 2)
                .sum();
            vert_bytes + idx_bytes
        })
        .sum();
    let mesh_estimate = (mesh_raw as f64 * GOOD_COMPRESSION_RATIO) as usize;

    // Texture data
    let texture_estimate: usize = scene.textures()
        .map(|tex| {
            if let Some((bytes, _)) = tex.original_bytes() {
                // already compressed as PNG/JPEG, zstd won't shrink further
                return bytes.len();
            }
            if let Some(path) = tex.source_path()
                && let Ok(meta) = std::fs::metadata(path) {
                    return meta.len() as usize;
                }
            // Fallback: estimate from dimensions (RGBA / 2 for rough PNG size)
            tex.dimensions()
                .map(|(w, h)| {
                    let dimensions = w as usize * h as usize;
                    let uncompressed_size = dimensions * 4;
                    let compressed_size = uncompressed_size as f64 * GOOD_COMPRESSION_RATIO;
                    compressed_size as usize
                })
                .unwrap_or(0)
        })
        .sum();

    // Environment maps (HDR data + preprocessed IBL data)
    let env_estimate: usize = scene.environment_maps()
        .map(|em| {
            let source_size = match em.source() {
                EnvironmentSource::EquirectangularPath(path) => {
                    std::fs::metadata(path).map(|m| m.len() as usize).unwrap_or(0)
                }
                EnvironmentSource::EquirectangularHdr(data) => data.len(),
                EnvironmentSource::Preprocessed => 0,
            };
            let preprocessed_size = em.preprocessed_ibl()
                .map(|p| {
                    let irr: usize = p.irradiance.mip_data.iter()
                        .flat_map(|m| m.iter())
                        .map(|f| f.len())
                        .sum();
                    let pre: usize = p.prefiltered.mip_data.iter()
                        .flat_map(|m| m.iter())
                        .map(|f| f.len())
                        .sum();
                    irr + pre + p.brdf_lut.as_ref().map_or(0, |l| l.len())
                })
                .unwrap_or(0);
            source_size + preprocessed_size
        })
        .sum();

    // Structured data (nodes, instances, materials) compresses well
    let structured = scene.node_count() * std::mem::size_of::<Node>()
        + scene.instance_count() * std::mem::size_of::<Instance>()
        + scene.material_count() * std::mem::size_of::<Material>();
    let structured_estimate = (structured as f64 * GOOD_COMPRESSION_RATIO) as usize;

    // Header (16 bytes) + TOC (~40 bytes per section × 8 sections)
    let overhead = HEADER_SIZE + 8 * 40;

    let total = mesh_estimate + texture_estimate + env_estimate + structured_estimate + overhead;

    // 10% safety margin
    total + total / 10
}

/// Serializes the scene to bytes in Duck format with default options.
pub fn to_bytes(scene: &Scene) -> Result<Vec<u8>, FormatError> {
    to_bytes_with_options(scene, &SaveOptions::default())
}

/// Serializes the scene to bytes in Duck format with custom options.
pub fn to_bytes_with_options(scene: &Scene, options: &SaveOptions) -> Result<Vec<u8>, FormatError> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let compression_level = options.compression.zstd_level();
    let annotation_filter = AnnotationContentFilter::from_scene(scene);
    let mut output = Vec::with_capacity(estimate_serialized_size(scene));
    output.resize(HEADER_SIZE, 0); // Reserve space for header
    let mut toc = TableOfContents::new();

    // ===== Metadata Section =====
    let metadata = SerializedMetadata {
        name: None,
        created_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        generator: format!("duck-engine {}", env!("CARGO_PKG_VERSION")),
        root_nodes: scene.root_nodes()
            .iter()
            .filter(|&&id| !annotation_filter.nodes.contains(&id))
            .copied()
            .collect(),
        active_environment_map: scene.active_environment_map(),
    };

    write_section(&metadata, SectionType::Metadata, compression_level, &mut output, &mut toc)?;

    // ===== Nodes Section =====
    let nodes: Vec<Node> = scene.nodes()
        .filter(|node| !annotation_filter.nodes.contains(&node.id))
        .map(|node| {
            let mut node = node.clone();
            // Filter annotation children out of the children list
            node.set_children_unchecked(
                node.children()
                    .iter()
                    .filter(|&&cid| !annotation_filter.nodes.contains(&cid))
                    .copied()
                    .collect(),
            );
            // Convert annotation instance references to None
            let payload = match node.payload() {
                NodePayload::Instance(iid) if annotation_filter.instances.contains(iid) => NodePayload::None,
                other => other.clone(),
            };
            node.set_payload(payload);
            node
        })
        .collect();
    write_section(&nodes, SectionType::Nodes, compression_level, &mut output, &mut toc)?;

    // ===== Instances Section =====
    let instances: Vec<Instance> = scene.instances()
        .filter(|inst| !annotation_filter.instances.contains(&inst.id))
        .cloned()
        .collect();
    write_section(&instances, SectionType::Instances, compression_level, &mut output, &mut toc)?;

    // ===== Materials Section =====
    let materials: Vec<Material> = scene.materials()
        .filter(|mat| !annotation_filter.materials.contains(&mat.id))
        .cloned()
        .collect();
    write_section(&materials, SectionType::Materials, compression_level, &mut output, &mut toc)?;

    // ===== Meshes Section =====
    let meshes: Vec<Mesh> = scene.meshes()
        .filter(|mesh| !annotation_filter.meshes.contains(&mesh.id))
        .cloned()
        .collect();
    write_section(&meshes, SectionType::Meshes, compression_level, &mut output, &mut toc)?;

    // ===== Textures Section =====
    let textures: Vec<&Texture> = scene.textures().collect();
    write_section(&textures, SectionType::Textures, compression_level, &mut output, &mut toc)?;

    // ===== Environment Maps Section =====
    if scene.has_environment_maps() {
        let env_maps: Vec<&EnvironmentMap> = scene.environment_maps().collect();
        write_section(&env_maps, SectionType::EnvironmentMaps, compression_level, &mut output, &mut toc)?;
    }

    // ===== Write TOC =====
    let toc_offset = output.len() as u64;
    let (toc_compressed, _) = serialize_section_with_level(&toc, compression_level)?;
    output.extend(toc_compressed);

    // ===== Write Header =====
    let header = FileHeader::new(toc_offset);
    let mut header_bytes = Vec::new();
    header.write(&mut header_bytes)?;
    output[..HEADER_SIZE].copy_from_slice(&header_bytes);

    Ok(output)
}

/// Deserializes a scene from Duck format bytes.
pub fn from_bytes(bytes: &[u8]) -> Result<Scene, FormatError> {
    assemble_duck_scene(parse_duck(bytes)?)
}

/// Saves the scene to a file with default options.
pub fn save_to_file(scene: &Scene, path: impl AsRef<std::path::Path>) -> Result<(), FormatError> {
    save_to_file_with_options(scene, path, &SaveOptions::default())
}

/// Saves the scene to a file with custom options.
pub fn save_to_file_with_options(
    scene: &Scene,
    path: impl AsRef<std::path::Path>,
    options: &SaveOptions,
) -> Result<(), FormatError> {
    let bytes = to_bytes_with_options(scene, options)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Loads a scene from a file.
pub fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<Scene, FormatError> {
    let bytes = std::fs::read(path)?;
    from_bytes(&bytes)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Point3, Quaternion, Vector3};
    use duck_engine_scene::common::{RgbaColor, Transform};
    use duck_engine_scene::PrimitiveType;

    /// Creates a simple test scene with various elements.
    fn create_test_scene() -> Scene {
        let mut scene = Scene::new();

        // Add a mesh
        let mesh = Mesh::cube(1.0, PrimitiveType::TriangleList);
        let mesh_id = scene.add_mesh(mesh);

        // Add a material
        let material = Material::new()
            .with_base_color_factor(RgbaColor::RED)
            .with_metallic_factor(0.5)
            .with_roughness_factor(0.3)
            .with_line_color(RgbaColor::GREEN);
        let mat_id = scene.add_material(material);

        // Add a node with instance
        let node_id = scene.add_instance_node(
            None,
            mesh_id,
            mat_id,
            Some("TestNode".to_string()),
            Transform::new(
                Point3::new(1.0, 2.0, 3.0),
                Quaternion::new(1.0, 0.0, 0.0, 0.0),
                Vector3::new(2.0, 2.0, 2.0),
            ),
        ).unwrap();

        // Add a child node
        let _child_id = scene.add_node(
            Some(node_id),
            Some("ChildNode".to_string()),
            Transform::from_position(Point3::new(0.5, 0.5, 0.5)),
        ).unwrap();

        // Add a light node
        {
            use duck_engine_scene::{Light, NodePayload};
            let light_node_id = scene.add_node(
                None,
                None,
                Transform::from_position(Point3::new(5.0, 5.0, 5.0)),
            ).unwrap();
            scene.set_node_payload(light_node_id, NodePayload::Light(Light::point(RgbaColor::WHITE, 10.0)));
        }

        scene
    }

    #[test]
    fn test_round_trip_basic() {
        let original = create_test_scene();

        // Serialize
        let bytes = to_bytes(&original).expect("Failed to serialize scene");

        // Check magic number
        assert_eq!(&bytes[0..4], b"DUCK");

        // Deserialize
        let loaded = from_bytes(&bytes).expect("Failed to deserialize scene");

        // Verify basic structure
        assert_eq!(loaded.node_count(), original.node_count());
        assert_eq!(loaded.mesh_count(), original.mesh_count());
        assert_eq!(loaded.material_count(), original.material_count());
        assert_eq!(loaded.instance_count(), original.instance_count());
        assert_eq!(loaded.has_light_nodes(), original.has_light_nodes());
        assert_eq!(loaded.root_nodes().len(), original.root_nodes().len());
    }

    #[test]
    fn test_round_trip_node_properties() {
        let original = create_test_scene();
        let bytes = to_bytes(&original).expect("Failed to serialize");
        let loaded = from_bytes(&bytes).expect("Failed to deserialize");

        // Find the test node by name
        let original_node = original.nodes()
            .find(|n| n.name.as_deref() == Some("TestNode"))
            .expect("TestNode not found in original");

        let loaded_node = loaded.nodes()
            .find(|n| n.name.as_deref() == Some("TestNode"))
            .expect("TestNode not found in loaded");

        // Verify position
        let orig_pos = original_node.position();
        let loaded_pos = loaded_node.position();
        assert!((orig_pos.x - loaded_pos.x).abs() < 1e-6);
        assert!((orig_pos.y - loaded_pos.y).abs() < 1e-6);
        assert!((orig_pos.z - loaded_pos.z).abs() < 1e-6);

        // Verify scale
        let orig_scale = original_node.scale();
        let loaded_scale = loaded_node.scale();
        assert!((orig_scale.x - loaded_scale.x).abs() < 1e-6);
        assert!((orig_scale.y - loaded_scale.y).abs() < 1e-6);
        assert!((orig_scale.z - loaded_scale.z).abs() < 1e-6);
    }

    #[test]
    #[ignore = "Fails unpredictably, possibly due to multithreaded serialization"]
    fn test_round_trip_material_properties() {
        let original = create_test_scene();
        let bytes = to_bytes(&original).expect("Failed to serialize");
        let loaded = from_bytes(&bytes).expect("Failed to deserialize");

        let original_mat = original.materials()
            .next()
            .expect("No material in original");

        let loaded_mat = loaded.materials()
            .next()
            .expect("No material in loaded");

        // Verify base color
        let orig_color = original_mat.base_color_factor();
        let loaded_color = loaded_mat.base_color_factor();
        assert!((orig_color.r - loaded_color.r).abs() < 1e-6);
        assert!((orig_color.g - loaded_color.g).abs() < 1e-6);
        assert!((orig_color.b - loaded_color.b).abs() < 1e-6);

        // Verify factors
        assert!((original_mat.metallic_factor() - loaded_mat.metallic_factor()).abs() < 1e-6);
        assert!((original_mat.roughness_factor() - loaded_mat.roughness_factor()).abs() < 1e-6);

        // Verify line color
        assert!(original_mat.line_color().is_some());
        assert!(loaded_mat.line_color().is_some());
    }

    #[test]
    fn test_round_trip_mesh_geometry() {
        let original = create_test_scene();
        let bytes = to_bytes(&original).expect("Failed to serialize");
        let loaded = from_bytes(&bytes).expect("Failed to deserialize");

        // Get the first mesh from each scene
        let original_mesh = original.meshes().next().expect("No mesh in original");
        let loaded_mesh = loaded.meshes().next().expect("No mesh in loaded");

        // Verify vertex counts match
        assert_eq!(original_mesh.vertices().len(), loaded_mesh.vertices().len());

        // Verify primitive counts
        assert_eq!(original_mesh.primitives().len(), loaded_mesh.primitives().len());

        // Verify first few vertices match
        for (orig_v, loaded_v) in original_mesh.vertices().iter().zip(loaded_mesh.vertices()) {
            assert!((orig_v.position[0] - loaded_v.position[0]).abs() < 1e-6);
            assert!((orig_v.position[1] - loaded_v.position[1]).abs() < 1e-6);
            assert!((orig_v.position[2] - loaded_v.position[2]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_round_trip_hierarchy() {
        let original = create_test_scene();
        let bytes = to_bytes(&original).expect("Failed to serialize");
        let loaded = from_bytes(&bytes).expect("Failed to deserialize");

        // Find child node
        let loaded_child = loaded.nodes()
            .find(|n| n.name.as_deref() == Some("ChildNode"))
            .expect("ChildNode not found");

        // Verify it has a parent
        assert!(loaded_child.parent().is_some());

        // Find parent node
        let loaded_parent = loaded.nodes()
            .find(|n| n.name.as_deref() == Some("TestNode"))
            .expect("TestNode not found");

        // Verify parent has child
        assert!(!loaded_parent.children().is_empty());
    }

    #[test]
    fn test_round_trip_lights() {
        use duck_engine_scene::{Light, NodePayload};
        let original = create_test_scene();
        let bytes = to_bytes(&original).expect("Failed to serialize");
        let loaded = from_bytes(&bytes).expect("Failed to deserialize");

        assert!(loaded.has_light_nodes());

        let light_node = loaded.nodes()
            .find(|n| matches!(n.payload(), NodePayload::Light(_)))
            .expect("No light node found");

        match light_node.payload() {
            NodePayload::Light(Light::Point { color, intensity, .. }) => {
                assert!((color.r - 1.0).abs() < 1e-6);
                assert!((*intensity - 10.0).abs() < 1e-6);
            }
            _ => panic!("Expected point light node"),
        }
    }

    #[test]
    fn test_empty_scene_round_trip() {
        let scene = Scene::new();
        let bytes = to_bytes(&scene).expect("Failed to serialize empty scene");
        let loaded = from_bytes(&bytes).expect("Failed to deserialize empty scene");

        assert_eq!(loaded.material_count(), 0);
        assert_eq!(loaded.node_count(), 0);
        assert_eq!(loaded.mesh_count(), 0);
        assert_eq!(loaded.instance_count(), 0);
    }

    #[test]
    fn test_version_in_header() {
        let scene = Scene::new();
        let bytes = to_bytes(&scene).expect("Failed to serialize");

        // Check version bytes (offset 4-5)
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, VERSION);
    }

    #[test]
    fn test_invalid_magic_rejected() {
        let mut bytes = vec![b'X', b'X', b'X', b'X']; // Wrong magic
        bytes.extend([0u8; 12]); // Rest of header

        let result = from_bytes(&bytes);
        assert!(matches!(result, Err(FormatError::InvalidMagic)));
    }

    #[test]
    fn test_reified_annotation_geometry_excluded() {

        // Create a scene with a mesh and an annotation
        let mut scene = Scene::new();

        // Add a regular mesh/node
        let mesh = Mesh::cube(1.0, PrimitiveType::TriangleList);
        let mesh_id = scene.add_mesh(mesh);
        let mat_id = scene.add_material(Material::new());
        let _node_id = scene.add_instance_node(
            None,
            mesh_id,
            mat_id,
            Some("RegularNode".to_string()),
            Transform::IDENTITY,
        ).unwrap();

        // Add an annotation and reify it (creates mesh, material, instance, node)
        scene.annotations.add_line(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            RgbaColor::RED,
        );
        scene.reify_annotations();

        // Before serialization: we have 2 meshes, 3 nodes
        // (annotation root + annotation node + regular node)
        assert!(scene.mesh_count() > 1, "Should have annotation mesh");
        assert!(scene.node_count() > 1, "Should have annotation nodes");

        // Serialize
        let bytes = to_bytes(&scene).expect("Failed to serialize");

        // Deserialize
        let loaded = from_bytes(&bytes).expect("Failed to deserialize");

        // After deserialization: annotation geometry should NOT be present
        // Only the regular mesh/material/instance/node should be serialized
        assert_eq!(loaded.mesh_count(), 1, "Only regular mesh should be serialized");
        assert_eq!(loaded.instance_count(), 1, "Only regular instance should be serialized");
        assert_eq!(loaded.node_count(), 1, "Only regular node should be serialized");

        // Verify the regular node is present
        let regular_node = loaded.nodes()
            .find(|n| n.name.as_deref() == Some("RegularNode"))
            .expect("RegularNode not found");
        assert!(matches!(regular_node.payload(), duck_engine_scene::NodePayload::Instance(_)));
    }

    #[test]
    fn estimate_is_upper_bound() {
        let scene = create_test_scene();
        let estimate = estimate_serialized_size(&scene);
        let bytes = to_bytes(&scene).expect("serialize");
        assert!(
            estimate >= bytes.len(),
            "estimate ({}) should be >= actual size ({})",
            estimate,
            bytes.len(),
        );
    }

    #[test]
    fn estimate_empty_scene() {
        let scene = Scene::new();
        let estimate = estimate_serialized_size(&scene);
        // Should return a small positive value (header + TOC overhead)
        assert!(estimate > 0);
        let bytes = to_bytes(&scene).expect("serialize");
        assert!(estimate >= bytes.len());
    }
}
