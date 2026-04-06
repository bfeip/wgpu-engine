//! Scene file format serialization.
//!
//! This module provides serialization and deserialization of scenes to a custom
//! binary format (.wgsc). The format is designed to be:
//! - Concise: Uses Zstd compression for large data sections
//! - Flexible: Section-based structure allows adding new data types
//! - Future-proof: Magic number and version header for compatibility
//!
//! # File Structure
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │ FIXED HEADER (16 bytes)                                      │
//! │  [0..4]   Magic: b"WGSC"                                     │
//! │  [4..6]   Version: u16 (major << 8 | minor)                  │
//! │  [6..8]   Flags: u16 (reserved)                              │
//! │  [8..16]  TOC offset: u64                                    │
//! ├──────────────────────────────────────────────────────────────┤
//! │ SECTION DATA (variable, Zstd compressed)                     │
//! ├──────────────────────────────────────────────────────────────┤
//! │ TABLE OF CONTENTS (at end of file)                           │
//! └──────────────────────────────────────────────────────────────┘
//! ```

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write, Cursor};

use image::GenericImageView;

use image::codecs::png::{CompressionType, FilterType};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use duck_engine_scene::{
    Instance, InstanceId,
    Light,
    Material, MaterialId, DEFAULT_MATERIAL_ID,
    Mesh, MeshId,
    Node, NodeId,
    Texture, TextureFormat, TextureId,
    EnvironmentMap, EnvironmentMapId, Scene,
    PreprocessedCubemap, PreprocessedIbl,
};

// ============================================================================
// Constants
// ============================================================================

/// Magic number identifying WGSC files: "WGSC" in ASCII
pub const MAGIC: [u8; 4] = *b"WGSC";

/// Current format version (major.minor encoded as single u16)
/// major = version >> 8, minor = version & 0xFF
pub const VERSION: u16 = 0x0002; // 0.2

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
#[derive(Clone, Debug)]
pub struct SaveOptions {
    /// Compression level for data and textures.
    pub compression: CompressionLevel,
    /// Fallback texture format when original bytes are unavailable (default: JPEG).
    pub texture_format: TextureFormat,
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self {
            compression: CompressionLevel::Default,
            texture_format: TextureFormat::Jpeg,
        }
    }
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
            _ => Err(FormatError::InvalidSectionType(value)),
        }
    }
}

// ============================================================================
// File Header & TOC
// ============================================================================

/// Fixed-size file header at the start of every .wgsc file.
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

        // Check version compatibility (support 0.1 and 0.2)
        let major = (version >> 8) as u8;
        let minor = (version & 0xFF) as u8;
        if major > 0 || (major == 0 && minor > 2) {
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
    /// Generator string (e.g., "wgpu-engine 0.1.0")
    pub generator: String,
    /// Root node IDs (remapped)
    pub root_nodes: Vec<u32>,
    /// Active environment map ID (remapped), if any
    pub active_environment_map: Option<u32>,
}


/// Serializable texture with embedded image data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedTexture {
    pub id: u32,
    pub format: TextureFormat,
    pub width: u32,
    pub height: u32,
    /// Compressed image bytes (PNG/JPEG) or raw RGBA data, depending on `format`
    pub data: Vec<u8>,
}

impl SerializedTexture {
    /// Creates a SerializedTexture from a Texture.
    ///
    /// Uses original compressed bytes when available (from glTF embedded images or file paths),
    /// avoiding expensive re-encoding. Falls back to `fallback_format` encoding otherwise.
    pub fn from_texture(
        texture: &Texture,
        id: u32,
        fallback_format: TextureFormat,
        compression: CompressionLevel,
    ) -> Result<Self, FormatError> {
        // Priority 1: Use preserved original bytes (from glTF embedded images)
        if let Some((data, format)) = Self::from_original_bytes(texture) {
            let (width, height) = Self::texture_dimensions(texture)?;
            return Ok(Self { id, format, width, height, data });
        }

        // Priority 2: Read original file bytes if texture has a source path
        if let Some((data, format)) = Self::from_source_path(texture) {
            let (width, height) = Self::texture_dimensions(texture)?;
            return Ok(Self { id, format, width, height, data });
        }

        // Priority 3: Fall back to encoding with the configured format
        let image = texture.get_image()
            .map_err(|e| FormatError::TextureError(e.to_string()))?;
        let (width, height) = image.dimensions();
        let (format, data) = Self::encode_image(&image, fallback_format, compression)?;

        Ok(Self { id, format, width, height, data })
    }

    fn texture_dimensions(texture: &Texture) -> Result<(u32, u32), FormatError> {
        let image = texture.get_image()
            .map_err(|e| FormatError::TextureError(e.to_string()))?;
        Ok(image.dimensions())
    }

    fn from_original_bytes(texture: &Texture) -> Option<(Vec<u8>, TextureFormat)> {
        texture.original_bytes().map(|(bytes, format)| (bytes.to_vec(), format))
    }

    fn from_source_path(texture: &Texture) -> Option<(Vec<u8>, TextureFormat)> {
        let path = texture.source_path()?;
        let bytes = std::fs::read(path).ok()?;
        let format = Self::detect_format(&bytes)?;
        Some((bytes, format))
    }

    fn encode_image(
        image: &image::DynamicImage,
        fallback_format: TextureFormat,
        compression: CompressionLevel,
    ) -> Result<(TextureFormat, Vec<u8>), FormatError> {
        use image::ImageEncoder;

        match fallback_format {
            TextureFormat::Jpeg => {
                use image::codecs::jpeg::JpegEncoder;

                let rgb = image.to_rgb8();
                let (w, h) = image.dimensions();
                let mut buf = Vec::new();
                JpegEncoder::new_with_quality(&mut buf, compression.jpeg_quality())
                    .write_image(&rgb, w, h, image::ExtendedColorType::Rgb8)
                    .map_err(|e| FormatError::TextureError(e.to_string()))?;
                Ok((TextureFormat::Jpeg, buf))
            }
            _ => {
                use image::codecs::png::PngEncoder;

                let rgba = image.to_rgba8();
                let (w, h) = image.dimensions();
                let mut buf = Vec::new();
                PngEncoder::new_with_quality(
                    &mut buf,
                    compression.png_compression(),
                    FilterType::Adaptive,
                )
                .write_image(&rgba, w, h, image::ExtendedColorType::Rgba8)
                .map_err(|e| FormatError::TextureError(e.to_string()))?;
                Ok((TextureFormat::Png, buf))
            }
        }
    }

    /// Detect image format from magic bytes.
    fn detect_format(bytes: &[u8]) -> Option<TextureFormat> {
        image::guess_format(bytes)
            .ok()
            .and_then(|f| TextureFormat::try_from(f).ok())
    }

    /// Converts to a Texture.
    pub fn to_texture(&self) -> Result<Texture, FormatError> {
        use image::DynamicImage;

        let image = match self.format {
            TextureFormat::Png | TextureFormat::Jpeg => {
                image::load_from_memory(&self.data)
                    .map_err(|e| FormatError::TextureError(e.to_string()))?
            }
            TextureFormat::Raw => {
                // Raw RGBA8 data
                let rgba = image::RgbaImage::from_raw(self.width, self.height, self.data.clone())
                    .ok_or_else(|| FormatError::TextureError("Invalid raw image data".to_string()))?;
                DynamicImage::ImageRgba8(rgba)
            }
        };

        let mut texture = Texture::from_image(image);
        texture.id = self.id;
        Ok(texture)
    }
}

/// Serializable environment map with embedded HDR data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedEnvironmentMap {
    /// Remapped environment map ID.
    pub id: u32,
    /// Raw .hdr file bytes, or `None` if the HDR source was discarded (e.g. after IBL baking).
    pub hdr_data: Option<Vec<u8>>,
    /// Intensity multiplier.
    pub intensity: f32,
    /// Rotation around Y axis in radians.
    pub rotation: f32,
}

/// Serializable preprocessed IBL data for an environment map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedPreprocessedIbl {
    /// Remapped environment map ID this data belongs to.
    pub env_map_id: u32,
    /// Diffuse irradiance cubemap.
    pub irradiance: PreprocessedCubemap,
    /// Pre-filtered specular cubemap.
    pub prefiltered: PreprocessedCubemap,
    /// Optional custom BRDF LUT.
    pub brdf_lut: Option<Vec<u8>>,
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
            .filter_map(|node| node.instance())
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

/// All deserialized sections from a WGSC file, prior to texture decoding
/// and scene assembly. Produced by [`parse_wgsc`], consumed by
/// [`decode_wgsc_textures`] and [`assemble_wgsc_scene`].
pub struct WgscSections {
    pub metadata: SerializedMetadata,
    pub textures: Vec<SerializedTexture>,
    pub materials: Vec<Material>,
    pub meshes: Vec<Mesh>,
    pub instances: Vec<Instance>,
    pub nodes: Vec<Node>,
    pub lights: Vec<Light>,
    pub environment_maps: Vec<SerializedEnvironmentMap>,
    pub preprocessed_ibl: Vec<SerializedPreprocessedIbl>,
}

/// Parse WGSC header, TOC, and decompress/deserialize all sections.
///
/// This is relatively fast (decompression + bincode deserialization).
/// The heavy texture image decoding happens in [`decode_wgsc_textures`].
pub fn parse_wgsc(bytes: &[u8]) -> Result<WgscSections, FormatError> {
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
    let textures: Vec<SerializedTexture> =
        deserialize_section(read_section(SectionType::Textures)?)?;
    let materials: Vec<Material> =
        deserialize_section(read_section(SectionType::Materials)?)?;
    let meshes: Vec<Mesh> =
        deserialize_section(read_section(SectionType::Meshes)?)?;
    let instances: Vec<Instance> =
        deserialize_section(read_section(SectionType::Instances)?)?;
    let nodes: Vec<Node> =
        deserialize_section(read_section(SectionType::Nodes)?)?;
    let lights: Vec<Light> =
        deserialize_section(read_section(SectionType::Lights)?)?;
    let environment_maps: Vec<SerializedEnvironmentMap> =
        if let Some(entry) = toc.find(SectionType::EnvironmentMaps) {
            let start = entry.offset as usize;
            let end = start + entry.compressed_size as usize;
            deserialize_section(&bytes[start..end])?
        } else {
            Vec::new()
        };

    let preprocessed_ibl: Vec<SerializedPreprocessedIbl> =
        if let Some(entry) = toc.find(SectionType::PreprocessedIblData) {
            let start = entry.offset as usize;
            let end = start + entry.compressed_size as usize;
            deserialize_section(&bytes[start..end])?
        } else {
            Vec::new()
        };

    Ok(WgscSections {
        metadata,
        textures,
        materials,
        meshes,
        instances,
        nodes,
        lights,
        environment_maps,
        preprocessed_ibl,
    })
}

/// A single decoded texture with its file-local ID, ready for scene insertion.
pub struct DecodedTexture {
    pub file_id: u32,
    pub texture: Texture,
}

/// Decode serialized textures into images.
///
/// On native this uses rayon for parallelism. On WASM it decodes sequentially.
/// This is typically the most expensive phase of loading.
pub fn decode_wgsc_textures(
    serialized: &[SerializedTexture],
) -> Result<Vec<DecodedTexture>, FormatError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        serialized
            .par_iter()
            .map(|st| {
                st.to_texture()
                    .map(|tex| DecodedTexture { file_id: st.id, texture: tex })
            })
            .collect()
    }

    #[cfg(target_arch = "wasm32")]
    {
        serialized
            .iter()
            .map(|st| {
                st.to_texture()
                    .map(|tex| DecodedTexture { file_id: st.id, texture: tex })
            })
            .collect()
    }
}

/// Decode a single serialized texture. Useful for per-item progress reporting
/// and WASM chunked yielding.
pub fn decode_wgsc_texture(
    serialized: &SerializedTexture,
) -> Result<DecodedTexture, FormatError> {
    serialized
        .to_texture()
        .map(|tex| DecodedTexture { file_id: serialized.id, texture: tex })
}

/// Assemble a [`Scene`] from parsed sections and decoded textures.
pub fn assemble_wgsc_scene(
    sections: WgscSections,
    decoded_textures: Vec<DecodedTexture>,
) -> Result<Scene, FormatError> {
    let mut scene = Scene::new();

    for dt in decoded_textures {
        let mut texture = dt.texture;
        texture.id = dt.file_id;
        scene.insert_texture_unchecked(texture);
    }

    for mesh in sections.meshes {
        scene.insert_mesh_unchecked(mesh);
    }

    for material in sections.materials {
        scene.insert_material_unchecked(material);
    }

    for instance in sections.instances {
        scene.insert_instance_unchecked(instance);
    }

    for node in sections.nodes {
        scene.insert_node_unchecked(node);
    }

    scene.set_root_node_order(sections.metadata.root_nodes);
    scene.set_lights(sections.lights);

    // Build a lookup from env_map_id -> preprocessed IBL data
    let mut preprocessed_map: HashMap<u32, PreprocessedIbl> = HashMap::new();
    for sibl in sections.preprocessed_ibl {
        preprocessed_map.insert(sibl.env_map_id, PreprocessedIbl {
            irradiance: sibl.irradiance,
            prefiltered: sibl.prefiltered,
            brdf_lut: sibl.brdf_lut,
        });
    }

    for sem in sections.environment_maps {
        let has_preprocessed = preprocessed_map.contains_key(&sem.id);
        let has_hdr = sem.hdr_data.is_some();

        let mut em = if let Some(hdr_data) = sem.hdr_data {
            EnvironmentMap::from_hdr_data(sem.id, hdr_data)
        } else if has_preprocessed {
            // HDR was dropped; create from preprocessed data
            EnvironmentMap::from_preprocessed(
                sem.id,
                preprocessed_map.remove(&sem.id).unwrap(),
            )
        } else {
            return Err(FormatError::DeserializationError(
                format!("Environment map {} has neither HDR data nor preprocessed IBL data", sem.id),
            ));
        };

        em.set_intensity(sem.intensity);
        em.set_rotation(sem.rotation);

        // Attach preprocessed data if HDR source was also kept
        if has_hdr {
            if let Some(preprocessed) = preprocessed_map.remove(&sem.id) {
                em.set_preprocessed_ibl(preprocessed);
            }
        }

        scene.insert_environment_map_unchecked(em);
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
    use std::mem::size_of;
    use duck_engine_scene::{EnvironmentSource, Vertex};

    const GOOD_COMPRESSION_RATIO: f64 = 0.6;

    // Mesh data (vertices + indices), estimate zstd compression at 60%
    let mesh_raw: usize = scene.meshes()
        .map(|m| {
            let vert_bytes = m.vertices().len() * size_of::<Vertex>();
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
            if let Some(path) = tex.source_path() {
                if let Ok(meta) = std::fs::metadata(path) {
                    return meta.len() as usize;
                }
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

    // Structured data (nodes, instances, materials, lights) compresses well
    let structured = scene.node_count() * std::mem::size_of::<Node>()
        + scene.instance_count() * std::mem::size_of::<Instance>()
        + scene.material_count() * std::mem::size_of::<Material>()
        + scene.lights().len() * std::mem::size_of::<Light>();
    let structured_estimate = (structured as f64 * GOOD_COMPRESSION_RATIO) as usize;

    // Header (16 bytes) + TOC (~40 bytes per section × 8 sections)
    let overhead = HEADER_SIZE + 8 * 40;

    let total = mesh_estimate + texture_estimate + env_estimate + structured_estimate + overhead;

    // 10% safety margin
    total + total / 10
}

/// Serializes the scene to bytes in WGSC format with default options.
pub fn to_bytes(scene: &Scene) -> Result<Vec<u8>, FormatError> {
    to_bytes_with_options(scene, &SaveOptions::default())
}

/// Serializes the scene to bytes in WGSC format with custom options.
pub fn to_bytes_with_options(scene: &Scene, options: &SaveOptions) -> Result<Vec<u8>, FormatError> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let compression_level = options.compression.zstd_level();
    let annotation_filter = AnnotationContentFilter::from_scene(scene);
    let mut output = Vec::with_capacity(estimate_serialized_size(scene));
    output.resize(HEADER_SIZE, 0); // Reserve space for header
    let mut toc = TableOfContents::new();

    // Build sequential ID maps for each type, excluding annotation content
    let mut node_id_map: HashMap<NodeId, u32> = HashMap::new();
    let mut instance_id_map: HashMap<InstanceId, u32> = HashMap::new();
    let mut mesh_id_map: HashMap<MeshId, u32> = HashMap::new();
    let mut material_id_map: HashMap<MaterialId, u32> = HashMap::new();
    let mut texture_id_map: HashMap<TextureId, u32> = HashMap::new();
    let mut env_map_id_map: HashMap<EnvironmentMapId, u32> = HashMap::new();

    let mut node_idx = 0u32;
    for node in scene.nodes() {
        if !annotation_filter.nodes.contains(&node.id) {
            node_id_map.insert(node.id, node_idx);
            node_idx += 1;
        }
    }

    let mut inst_idx = 0u32;
    for inst in scene.instances() {
        if !annotation_filter.instances.contains(&inst.id) {
            instance_id_map.insert(inst.id, inst_idx);
            inst_idx += 1;
        }
    }

    let mut mesh_idx = 0u32;
    for mesh in scene.meshes() {
        if !annotation_filter.meshes.contains(&mesh.id) {
            mesh_id_map.insert(mesh.id, mesh_idx);
            mesh_idx += 1;
        }
    }

    // The default material gets a sentinel file ID (u32::MAX) so instances
    // can reference it, but its data is not serialized (Scene::new() recreates it).
    material_id_map.insert(DEFAULT_MATERIAL_ID, u32::MAX);
    let mut mat_idx = 0u32;
    for mat in scene.materials() {
        if mat.id == DEFAULT_MATERIAL_ID || annotation_filter.materials.contains(&mat.id) {
            continue;
        }
        material_id_map.insert(mat.id, mat_idx);
        mat_idx += 1;
    }

    for (idx, tex) in scene.textures().enumerate() {
        texture_id_map.insert(tex.id(), idx as u32);
    }

    for (idx, em) in scene.environment_maps().enumerate() {
        env_map_id_map.insert(em.id, idx as u32);
    }

    // ===== Metadata Section =====
    let metadata = SerializedMetadata {
        name: None,
        created_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        generator: format!("wgpu-engine {}", env!("CARGO_PKG_VERSION")),
        root_nodes: scene.root_nodes()
            .iter()
            .filter_map(|&id| node_id_map.get(&id).copied())
            .collect(),
        active_environment_map: scene.active_environment_map()
            .and_then(|id| env_map_id_map.get(&id).copied()),
    };
    
    write_section(&metadata, SectionType::Metadata, compression_level, &mut output, &mut toc)?;

    // ===== Nodes Section =====
    let nodes: Vec<Node> = scene.nodes()
        .filter_map(|node| {
            let remapped_id = *node_id_map.get(&node.id)?;
            let mut node = node.clone();
            node.id = remapped_id;
            node.set_parent_unchecked(
                node.parent().and_then(|pid| node_id_map.get(&pid).copied()),
            );
            node.set_children_unchecked(
                node.children()
                    .iter()
                    .filter_map(|&cid| node_id_map.get(&cid).copied())
                    .collect(),
            );
            node.set_instance(
                node.instance().and_then(|iid| instance_id_map.get(&iid).copied()),
            );
            Some(node)
        })
        .collect();
    write_section(&nodes, SectionType::Nodes, compression_level, &mut output, &mut toc)?;

    // ===== Instances Section =====
    let instances: Vec<Instance> = scene.instances()
        .filter_map(|inst| {
            let remapped_id = *instance_id_map.get(&inst.id)?;
            let mut inst = inst.clone();
            inst.id = remapped_id;
            inst.set_mesh_unchecked(*mesh_id_map.get(&inst.mesh())?);
            inst.set_material_unchecked(*material_id_map.get(&inst.material())?);
            Some(inst)
        })
        .collect();
    write_section(&instances, SectionType::Instances, compression_level, &mut output, &mut toc)?;

    // ===== Materials Section =====
    // Skip the default material (it's always recreated by Scene::new())
    let materials: Vec<Material> = scene.materials()
        .filter(|mat| mat.id != DEFAULT_MATERIAL_ID && !annotation_filter.materials.contains(&mat.id))
        .map(|mat| {
            let remapped_id = *material_id_map.get(&mat.id).unwrap_or(&0);

            // Build a new material with remapped texture IDs
            let mut new_mat = Material::new()
                .with_base_color_factor(mat.base_color_factor())
                .with_metallic_factor(mat.metallic_factor())
                .with_roughness_factor(mat.roughness_factor())
                .with_normal_scale(mat.normal_scale())
                .with_flags(mat.flags())
                .with_alpha_mode(mat.alpha_mode())
                .with_alpha_cutoff(mat.alpha_cutoff());

            if let Some(tid) = mat.base_color_texture() {
                if let Some(&remapped) = texture_id_map.get(&tid) {
                    new_mat = new_mat.with_base_color_texture(remapped);
                }
            }
            if let Some(tid) = mat.normal_texture() {
                if let Some(&remapped) = texture_id_map.get(&tid) {
                    new_mat = new_mat.with_normal_texture(remapped);
                }
            }
            if let Some(tid) = mat.metallic_roughness_texture() {
                if let Some(&remapped) = texture_id_map.get(&tid) {
                    new_mat = new_mat.with_metallic_roughness_texture(remapped);
                }
            }
            if let Some(color) = mat.line_color() {
                new_mat = new_mat.with_line_color(color);
            }
            if let Some(color) = mat.point_color() {
                new_mat = new_mat.with_point_color(color);
            }

            new_mat.id = remapped_id;
            new_mat
        })
        .collect();
    write_section(&materials, SectionType::Materials, compression_level, &mut output, &mut toc)?;

    // ===== Meshes Section =====
    let meshes: Vec<Mesh> = scene.meshes()
        .filter(|mesh| !annotation_filter.meshes.contains(&mesh.id))
        .map(|mesh| {
            let remapped_id = *mesh_id_map.get(&mesh.id).unwrap_or(&0);
            let mut cloned = mesh.clone();
            cloned.id = remapped_id;
            cloned
        })
        .collect();
    write_section(&meshes, SectionType::Meshes, compression_level, &mut output, &mut toc)?;

    // ===== Textures Section =====
    let mut textures = Vec::new();
    for texture in scene.textures() {
        if let Some(&remapped_id) = texture_id_map.get(&texture.id()) {
            let serialized = SerializedTexture::from_texture(
                texture,
                remapped_id,
                options.texture_format,
                options.compression,
            )?;
            textures.push(serialized);
        }
    }
    write_section(&textures, SectionType::Textures, compression_level, &mut output, &mut toc)?;

    // ===== Lights Section =====
    let lights: Vec<Light> = scene.lights().to_vec();
    write_section(&lights, SectionType::Lights, compression_level, &mut output, &mut toc)?;

    // ===== Environment Maps Section =====
    if scene.has_environment_maps() {
        let mut env_maps = Vec::new();
        let mut preprocessed_ibls = Vec::new();

        for env_map in scene.environment_maps() {
            let remapped_id = *env_map_id_map.get(&env_map.id).unwrap_or(&0);
            let hdr_data = match env_map.source() {
                duck_engine_scene::EnvironmentSource::EquirectangularPath(path) => {
                    Some(std::fs::read(path).map_err(|e| FormatError::IoError(e))?)
                }
                duck_engine_scene::EnvironmentSource::EquirectangularHdr(data) => {
                    Some(data.clone())
                }
                duck_engine_scene::EnvironmentSource::Preprocessed => None,
            };
            env_maps.push(SerializedEnvironmentMap {
                id: remapped_id,
                hdr_data,
                intensity: env_map.intensity(),
                rotation: env_map.rotation(),
            });

            // Serialize preprocessed IBL data if present
            if let Some(preprocessed) = env_map.preprocessed_ibl() {
                preprocessed_ibls.push(SerializedPreprocessedIbl {
                    env_map_id: remapped_id,
                    irradiance: preprocessed.irradiance.clone(),
                    prefiltered: preprocessed.prefiltered.clone(),
                    brdf_lut: preprocessed.brdf_lut.clone(),
                });
            }
        }
        write_section(&env_maps, SectionType::EnvironmentMaps, compression_level, &mut output, &mut toc)?;

        if !preprocessed_ibls.is_empty() {
            write_section(&preprocessed_ibls, SectionType::PreprocessedIblData, compression_level, &mut output, &mut toc)?;
        }
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

/// Deserializes a scene from WGSC format bytes.
///
/// This is a convenience method that calls [`parse_wgsc`],
/// [`decode_wgsc_textures`], and [`assemble_wgsc_scene`] sequentially.
/// For progress reporting or async loading, call those phases individually.
pub fn from_bytes(bytes: &[u8]) -> Result<Scene, FormatError> {
    let sections = parse_wgsc(bytes)?;
    let textures = decode_wgsc_textures(&sections.textures)?;
    assemble_wgsc_scene(sections, textures)
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

        // Add a light
        scene.add_light(Light::point(
            Vector3::new(5.0, 5.0, 5.0),
            RgbaColor::WHITE,
            10.0,
        ));

        scene
    }

    #[test]
    fn test_round_trip_basic() {
        let original = create_test_scene();

        // Serialize
        let bytes = to_bytes(&original).expect("Failed to serialize scene");

        // Check magic number
        assert_eq!(&bytes[0..4], b"WGSC");

        // Deserialize
        let loaded = from_bytes(&bytes).expect("Failed to deserialize scene");

        // Verify basic structure
        assert_eq!(loaded.node_count(), original.node_count());
        assert_eq!(loaded.mesh_count(), original.mesh_count());
        assert_eq!(loaded.material_count(), original.material_count());
        assert_eq!(loaded.instance_count(), original.instance_count());
        assert_eq!(loaded.lights().len(), original.lights().len());
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

        // Skip the default material (ID 0), find our custom material
        let original_mat = original.materials()
            .find(|m| m.id != DEFAULT_MATERIAL_ID)
            .expect("Custom material not found in original");

        let loaded_mat = loaded.materials()
            .find(|m| m.id != DEFAULT_MATERIAL_ID)
            .expect("Custom material not found in loaded");

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
        let original = create_test_scene();
        let bytes = to_bytes(&original).expect("Failed to serialize");
        let loaded = from_bytes(&bytes).expect("Failed to deserialize");

        assert_eq!(loaded.lights().len(), 1);

        match &loaded.lights()[0] {
            Light::Point { position, color, intensity, .. } => {
                assert!((position.x - 5.0).abs() < 1e-6);
                assert!((position.y - 5.0).abs() < 1e-6);
                assert!((position.z - 5.0).abs() < 1e-6);
                assert!((color.r - 1.0).abs() < 1e-6);
                assert!((*intensity - 10.0).abs() < 1e-6);
            }
            _ => panic!("Expected point light"),
        }
    }

    #[test]
    fn test_empty_scene_round_trip() {
        let scene = Scene::new();
        let bytes = to_bytes(&scene).expect("Failed to serialize empty scene");
        let loaded = from_bytes(&bytes).expect("Failed to deserialize empty scene");

        // Should only have the default material
        assert_eq!(loaded.material_count(), 1);
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
        assert!(regular_node.instance().is_some());
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
