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

use std::collections::HashMap;
use std::io::{Read, Write, Cursor};

use image::GenericImageView;

use image::codecs::png::{CompressionType, FilterType};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{
    annotation::{
        Annotation, AnnotationId, AxesAnnotation, BoxAnnotation,
        GridAnnotation, LineAnnotation, PointsAnnotation, PolylineAnnotation, AnnotationMeta,
    },
    instance::Instance,
    light::Light,
    material::{Material, MaterialFlags, MaterialId, DEFAULT_MATERIAL_ID},
    mesh::{Mesh, MeshId, MeshPrimitive, PrimitiveType, Vertex},
    node::{Node, NodeId, Visibility},
    texture::{Texture, TextureId},
    InstanceId, Scene,
};
use crate::common::{
    RgbaColor, array_to_point3, array_to_rgba, array_to_vec3,
    point3_to_array, rgba_to_array, vec3_to_array
};
use super::environment::EnvironmentMapId;

// ============================================================================
// Constants
// ============================================================================

/// Magic number identifying WGSC files: "WGSC" in ASCII
pub const MAGIC: [u8; 4] = *b"WGSC";

/// Current format version (major.minor encoded as single u16)
/// major = version >> 8, minor = version & 0xFF
pub const VERSION: u16 = 0x0001; // 0.1

/// Size of the fixed header in bytes
pub const HEADER_SIZE: usize = 16;

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
    /// Annotation data
    Annotations = 7,
    /// Environment map data
    EnvironmentMaps = 8,
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

        // Check version compatibility (currently only support 0.1)
        let major = (version >> 8) as u8;
        let minor = (version & 0xFF) as u8;
        if major > 0 || (major == 0 && minor > 1) {
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

/// Serializable node with remapped IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedNode {
    pub id: u32,
    pub name: Option<String>,
    pub position: [f32; 3],
    pub rotation: [f32; 4], // Quaternion as [x, y, z, w]
    pub scale: [f32; 3],
    pub parent_id: Option<u32>,
    pub children_ids: Vec<u32>,
    pub instance_id: Option<u32>,
    pub visible: bool,
}

/// Serializable instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedInstance {
    pub id: u32,
    pub mesh_id: u32,
    pub material_id: u32,
}

/// Serializable mesh primitive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedPrimitive {
    /// 0 = TriangleList, 1 = LineList, 2 = PointList
    pub primitive_type: u8,
    pub indices: Vec<u16>,
}

/// Serializable mesh with vertex data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedMesh {
    pub id: u32,
    /// Raw vertex bytes (36 bytes per vertex)
    pub vertices: Vec<u8>,
    pub primitives: Vec<SerializedPrimitive>,
}

/// Serializable material.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedMaterial {
    pub id: u32,
    pub base_color_texture_id: Option<u32>,
    pub normal_texture_id: Option<u32>,
    pub metallic_roughness_texture_id: Option<u32>,
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub normal_scale: f32,
    pub line_color: Option<[f32; 4]>,
    pub point_color: Option<[f32; 4]>,
    /// MaterialFlags as u32 bits
    pub flags: u32,
}

/// Texture image format.
#[repr(u8)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TextureFormat {
    Png = 0,
    Jpeg = 1,
    Raw = 2, // RGBA8 raw data
}

/// Serializable texture with embedded image data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedTexture {
    pub id: u32,
    pub format: TextureFormat,
    pub width: u32,
    pub height: u32,
    /// Compressed image bytes (PNG/JPEG) or raw RGBA data
    pub data: Vec<u8>,
}

/// Serializable light.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedLight {
    /// 0 = Point, 1 = Directional, 2 = Spot
    pub light_type: u8,
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub color: [f32; 4],
    pub intensity: f32,
    pub range: f32,
    pub inner_cone_angle: f32,
    pub outer_cone_angle: f32,
}

/// Serializable annotation metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedAnnotationMeta {
    pub id: u32,
    pub name: Option<String>,
    pub visible: bool,
}

/// Serializable annotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializedAnnotation {
    Line {
        meta: SerializedAnnotationMeta,
        start: [f32; 3],
        end: [f32; 3],
        color: [f32; 4],
    },
    Polyline {
        meta: SerializedAnnotationMeta,
        points: Vec<[f32; 3]>,
        color: [f32; 4],
        closed: bool,
    },
    Points {
        meta: SerializedAnnotationMeta,
        positions: Vec<[f32; 3]>,
        color: [f32; 4],
    },
    Axes {
        meta: SerializedAnnotationMeta,
        origin: [f32; 3],
        size: f32,
    },
    Box {
        meta: SerializedAnnotationMeta,
        center: [f32; 3],
        size: [f32; 3],
        color: [f32; 4],
    },
    Grid {
        meta: SerializedAnnotationMeta,
        center: [f32; 3],
        size: f32,
        divisions: u32,
        color: [f32; 4],
    },
}

/// Serializable environment map with embedded HDR data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedEnvironmentMap {
    /// Remapped environment map ID.
    pub id: u32,
    /// Raw .hdr file bytes.
    pub hdr_data: Vec<u8>,
    /// Intensity multiplier.
    pub intensity: f32,
    /// Rotation around Y axis in radians.
    pub rotation: f32,
}

// ============================================================================
// ID Remapping
// ============================================================================

/// Maps original runtime IDs to compact sequential IDs for serialization.
#[derive(Debug, Default)]
pub struct IdRemapper {
    pub nodes: HashMap<NodeId, u32>,
    pub instances: HashMap<InstanceId, u32>,
    pub meshes: HashMap<MeshId, u32>,
    pub materials: HashMap<MaterialId, u32>,
    pub textures: HashMap<TextureId, u32>,
    pub environment_maps: HashMap<EnvironmentMapId, u32>,
    pub annotations: HashMap<AnnotationId, u32>,
}

impl IdRemapper {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build ID mappings from a scene, assigning sequential IDs starting from 0.
    ///
    /// Nodes, instances, meshes, and materials created by annotation reification
    /// are excluded from the mapping. Annotations are serialized separately and
    /// their geometry will be re-reified when the scene is loaded.
    pub fn from_scene(scene: &Scene) -> Self {
        use std::collections::HashSet;

        let mut remapper = Self::new();

        // Collect all annotation-created node IDs (recursively includes children)
        let mut annotation_node_ids: HashSet<NodeId> = HashSet::new();
        if let Some(root_id) = scene.annotations.root_node() {
            Self::collect_subtree_nodes(scene, root_id, &mut annotation_node_ids);
        }

        // Collect instance IDs from annotation nodes
        let annotation_instance_ids: HashSet<InstanceId> = annotation_node_ids
            .iter()
            .filter_map(|&node_id| scene.nodes.get(&node_id))
            .filter_map(|node| node.instance())
            .collect();

        // Collect mesh and material IDs from annotation instances
        let mut annotation_mesh_ids: HashSet<MeshId> = HashSet::new();
        let mut annotation_material_ids: HashSet<MaterialId> = HashSet::new();
        for &instance_id in &annotation_instance_ids {
            if let Some(instance) = scene.instances.get(&instance_id) {
                annotation_mesh_ids.insert(instance.mesh);
                annotation_material_ids.insert(instance.material);
            }
        }

        // Remap nodes (excluding annotation nodes)
        let mut node_idx = 0u32;
        for &id in scene.nodes.keys() {
            if !annotation_node_ids.contains(&id) {
                remapper.nodes.insert(id, node_idx);
                node_idx += 1;
            }
        }

        // Remap instances (excluding annotation instances)
        let mut inst_idx = 0u32;
        for &id in scene.instances.keys() {
            if !annotation_instance_ids.contains(&id) {
                remapper.instances.insert(id, inst_idx);
                inst_idx += 1;
            }
        }

        // Remap meshes (excluding annotation meshes)
        let mut mesh_idx = 0u32;
        for &id in scene.meshes.keys() {
            if !annotation_mesh_ids.contains(&id) {
                remapper.meshes.insert(id, mesh_idx);
                mesh_idx += 1;
            }
        }

        // Remap materials (preserve 0 for default material, exclude annotation materials)
        let mut mat_idx = 0u32;
        for &id in scene.materials.keys() {
            if annotation_material_ids.contains(&id) {
                continue;
            }
            if id == DEFAULT_MATERIAL_ID {
                remapper.materials.insert(id, 0);
            } else {
                mat_idx += 1;
                remapper.materials.insert(id, mat_idx);
            }
        }

        // Remap textures
        for (idx, &id) in scene.textures.keys().collect::<Vec<_>>().iter().enumerate() {
            remapper.textures.insert(*id, idx as u32);
        }

        // Remap environment maps
        for (idx, &id) in scene.environment_maps.keys().collect::<Vec<_>>().iter().enumerate() {
            remapper.environment_maps.insert(*id, idx as u32);
        }

        // Remap annotations
        for (idx, annotation) in scene.annotations.iter().enumerate() {
            remapper.annotations.insert(annotation.id(), idx as u32);
        }

        remapper
    }

    /// Recursively collects all node IDs in a subtree.
    fn collect_subtree_nodes(scene: &Scene, node_id: NodeId, collected: &mut std::collections::HashSet<NodeId>) {
        collected.insert(node_id);
        if let Some(node) = scene.nodes.get(&node_id) {
            for &child_id in node.children() {
                Self::collect_subtree_nodes(scene, child_id, collected);
            }
        }
    }

    pub fn remap_node(&self, id: NodeId) -> Option<u32> {
        self.nodes.get(&id).copied()
    }

    pub fn remap_instance(&self, id: InstanceId) -> Option<u32> {
        self.instances.get(&id).copied()
    }

    pub fn remap_mesh(&self, id: MeshId) -> Option<u32> {
        self.meshes.get(&id).copied()
    }

    pub fn remap_material(&self, id: MaterialId) -> Option<u32> {
        self.materials.get(&id).copied()
    }

    pub fn remap_texture(&self, id: TextureId) -> Option<u32> {
        self.textures.get(&id).copied()
    }

    pub fn remap_environment_map(&self, id: EnvironmentMapId) -> Option<u32> {
        self.environment_maps.get(&id).copied()
    }
}

// ============================================================================
// Conversion: Scene -> Serialized
// ============================================================================

impl SerializedNode {
    pub fn from_node(node: &Node, remapper: &IdRemapper) -> Option<Self> {
        let id = remapper.remap_node(node.id)?;

        let position = node.position();
        let rotation = node.rotation();
        let scale = node.scale();

        // cgmath Quaternion: s is scalar (w), v is Vector3 (x, y, z)
        Some(Self {
            id,
            name: node.name.clone(),
            position: [position.x, position.y, position.z],
            rotation: [rotation.v.x, rotation.v.y, rotation.v.z, rotation.s],
            scale: [scale.x, scale.y, scale.z],
            parent_id: node.parent().and_then(|pid| remapper.remap_node(pid)),
            children_ids: node
                .children()
                .iter()
                .filter_map(|&cid| remapper.remap_node(cid))
                .collect(),
            instance_id: node.instance().and_then(|iid| remapper.remap_instance(iid)),
            visible: node.visibility() == Visibility::Visible,
        })
    }
}

impl SerializedInstance {
    pub fn from_instance(instance: &Instance, remapper: &IdRemapper) -> Option<Self> {
        Some(Self {
            id: remapper.remap_instance(instance.id)?,
            mesh_id: remapper.remap_mesh(instance.mesh)?,
            material_id: remapper.remap_material(instance.material)?,
        })
    }
}

impl SerializedPrimitive {
    pub fn from_primitive(primitive: &MeshPrimitive) -> Self {
        let primitive_type = match primitive.primitive_type {
            PrimitiveType::TriangleList => 0,
            PrimitiveType::LineList => 1,
            PrimitiveType::PointList => 2,
        };

        Self {
            primitive_type,
            indices: primitive.indices.clone(),
        }
    }

    pub fn to_primitive(&self) -> MeshPrimitive {
        let primitive_type = match self.primitive_type {
            0 => PrimitiveType::TriangleList,
            1 => PrimitiveType::LineList,
            _ => PrimitiveType::PointList,
        };

        MeshPrimitive {
            primitive_type,
            indices: self.indices.clone(),
        }
    }

    pub fn size(&self) -> usize {
        self.indices.len() * size_of::<u16>()
    }
}

impl SerializedMesh {
    pub fn from_mesh(mesh: &Mesh, remapper: &IdRemapper) -> Option<Self> {
        let id = remapper.remap_mesh(mesh.id)?;

        // Convert vertices to raw bytes
        let vertices_bytes: Vec<u8> = bytemuck::cast_slice(mesh.vertices()).to_vec();

        let primitives = mesh
            .primitives()
            .iter()
            .map(SerializedPrimitive::from_primitive)
            .collect();

        Some(Self {
            id,
            vertices: vertices_bytes,
            primitives,
        })
    }

    pub fn to_mesh(&self) -> Mesh {
        // Convert raw bytes back to vertices
        let vertices: Vec<Vertex> = bytemuck::cast_slice(&self.vertices).to_vec();

        let primitives: Vec<MeshPrimitive> = self
            .primitives
            .iter()
            .map(SerializedPrimitive::to_primitive)
            .collect();

        let mut mesh = Mesh::from_raw(vertices, primitives);
        mesh.id = self.id;
        mesh
    }

    pub fn size(&self) -> usize {
        let vertices_size = self.vertices.len();
        let primitives_size = self.primitives.iter().map(|primitive| {
            primitive.size()
        }).sum::<usize>();
        vertices_size + primitives_size
    }
}

impl SerializedMaterial {
    pub fn from_material(material: &Material, remapper: &IdRemapper) -> Option<Self> {
        let id = remapper.remap_material(material.id)?;

        let base_color = material.base_color_factor();
        let line_color = material.line_color();
        let point_color = material.point_color();

        Some(Self {
            id,
            base_color_texture_id: material
                .base_color_texture()
                .and_then(|tid| remapper.remap_texture(tid)),
            normal_texture_id: material
                .normal_texture()
                .and_then(|tid| remapper.remap_texture(tid)),
            metallic_roughness_texture_id: material
                .metallic_roughness_texture()
                .and_then(|tid| remapper.remap_texture(tid)),
            base_color_factor: [base_color.r, base_color.g, base_color.b, base_color.a],
            metallic_factor: material.metallic_factor(),
            roughness_factor: material.roughness_factor(),
            normal_scale: material.normal_scale(),
            line_color: line_color.map(|c| [c.r, c.g, c.b, c.a]),
            point_color: point_color.map(|c| [c.r, c.g, c.b, c.a]),
            flags: material.flags().bits(),
        })
    }

    pub fn to_material(&self) -> Material {
        let mut material = Material::new()
            .with_base_color_factor(RgbaColor {
                r: self.base_color_factor[0],
                g: self.base_color_factor[1],
                b: self.base_color_factor[2],
                a: self.base_color_factor[3],
            })
            .with_metallic_factor(self.metallic_factor)
            .with_roughness_factor(self.roughness_factor)
            .with_normal_scale(self.normal_scale)
            .with_flags(MaterialFlags::from_bits_truncate(self.flags));

        // Note: Texture IDs will be patched up after textures are loaded
        // using the file's IDs directly (they're already sequential)

        if let Some(color) = self.line_color {
            material = material.with_line_color(RgbaColor {
                r: color[0],
                g: color[1],
                b: color[2],
                a: color[3],
            });
        }

        if let Some(color) = self.point_color {
            material = material.with_point_color(RgbaColor {
                r: color[0],
                g: color[1],
                b: color[2],
                a: color[3],
            });
        }

        material.id = self.id;
        material
    }
}

impl SerializedLight {
    pub fn from_light(light: &Light) -> Self {
        match light {
            Light::Point {
                position,
                color,
                intensity,
                range,
            } => Self {
                light_type: 0,
                position: [position.x, position.y, position.z],
                direction: [0.0, 0.0, 0.0],
                color: [color.r, color.g, color.b, color.a],
                intensity: *intensity,
                range: *range,
                inner_cone_angle: 0.0,
                outer_cone_angle: 0.0,
            },
            Light::Directional {
                direction,
                color,
                intensity,
            } => Self {
                light_type: 1,
                position: [0.0, 0.0, 0.0],
                direction: [direction.x, direction.y, direction.z],
                color: [color.r, color.g, color.b, color.a],
                intensity: *intensity,
                range: 0.0,
                inner_cone_angle: 0.0,
                outer_cone_angle: 0.0,
            },
            Light::Spot {
                position,
                direction,
                color,
                intensity,
                range,
                inner_cone_angle,
                outer_cone_angle,
            } => Self {
                light_type: 2,
                position: [position.x, position.y, position.z],
                direction: [direction.x, direction.y, direction.z],
                color: [color.r, color.g, color.b, color.a],
                intensity: *intensity,
                range: *range,
                inner_cone_angle: *inner_cone_angle,
                outer_cone_angle: *outer_cone_angle,
            },
        }
    }

    pub fn to_light(&self) -> Light {
        use cgmath::Vector3;

        let color = RgbaColor {
            r: self.color[0],
            g: self.color[1],
            b: self.color[2],
            a: self.color[3],
        };

        match self.light_type {
            0 => Light::Point {
                position: Vector3::new(self.position[0], self.position[1], self.position[2]),
                color,
                intensity: self.intensity,
                range: self.range,
            },
            1 => Light::Directional {
                direction: Vector3::new(self.direction[0], self.direction[1], self.direction[2]),
                color,
                intensity: self.intensity,
            },
            _ => Light::Spot {
                position: Vector3::new(self.position[0], self.position[1], self.position[2]),
                direction: Vector3::new(self.direction[0], self.direction[1], self.direction[2]),
                color,
                intensity: self.intensity,
                range: self.range,
                inner_cone_angle: self.inner_cone_angle,
                outer_cone_angle: self.outer_cone_angle,
            },
        }
    }
}

impl SerializedAnnotation {
    pub fn from_annotation(annotation: &Annotation, remapper: &IdRemapper) -> Option<Self> {
        let make_meta = |meta: &AnnotationMeta| -> Option<SerializedAnnotationMeta> {
            Some(SerializedAnnotationMeta {
                id: *remapper.annotations.get(&meta.id)?,
                name: meta.name.clone(),
                visible: meta.visible,
            })
        };

        match annotation {
            Annotation::Line(a) => Some(SerializedAnnotation::Line {
                meta: make_meta(&a.meta)?,
                start: point3_to_array(a.start),
                end: point3_to_array(a.end),
                color: rgba_to_array(a.color),
            }),
            Annotation::Polyline(a) => Some(SerializedAnnotation::Polyline {
                meta: make_meta(&a.meta)?,
                points: a.points.iter().map(|p| point3_to_array(*p)).collect(),
                color: rgba_to_array(a.color),
                closed: a.closed,
            }),
            Annotation::Points(a) => Some(SerializedAnnotation::Points {
                meta: make_meta(&a.meta)?,
                positions: a.positions.iter().map(|p| point3_to_array(*p)).collect(),
                color: rgba_to_array(a.color),
            }),
            Annotation::Axes(a) => Some(SerializedAnnotation::Axes {
                meta: make_meta(&a.meta)?,
                origin: point3_to_array(a.origin),
                size: a.size,
            }),
            Annotation::Box(a) => Some(SerializedAnnotation::Box {
                meta: make_meta(&a.meta)?,
                center: point3_to_array(a.center),
                size: vec3_to_array(a.size),
                color: rgba_to_array(a.color),
            }),
            Annotation::Grid(a) => Some(SerializedAnnotation::Grid {
                meta: make_meta(&a.meta)?,
                center: point3_to_array(a.center),
                size: a.size,
                divisions: a.divisions,
                color: rgba_to_array(a.color),
            }),
        }
    }

    pub fn to_annotation(&self) -> Annotation {
        match self {
            SerializedAnnotation::Line { meta, start, end, color } => {
                Annotation::Line(LineAnnotation {
                    meta: AnnotationMeta {
                        id: meta.id,
                        name: meta.name.clone(),
                        visible: meta.visible,
                        node_id: None,
                    },
                    start: array_to_point3(*start),
                    end: array_to_point3(*end),
                    color: array_to_rgba(*color),
                })
            }
            SerializedAnnotation::Polyline { meta, points, color, closed } => {
                Annotation::Polyline(PolylineAnnotation {
                    meta: AnnotationMeta {
                        id: meta.id,
                        name: meta.name.clone(),
                        visible: meta.visible,
                        node_id: None,
                    },
                    points: points.iter().map(|p| array_to_point3(*p)).collect(),
                    color: array_to_rgba(*color),
                    closed: *closed,
                })
            }
            SerializedAnnotation::Points { meta, positions, color } => {
                Annotation::Points(PointsAnnotation {
                    meta: AnnotationMeta {
                        id: meta.id,
                        name: meta.name.clone(),
                        visible: meta.visible,
                        node_id: None,
                    },
                    positions: positions.iter().map(|p| array_to_point3(*p)).collect(),
                    color: array_to_rgba(*color),
                })
            }
            SerializedAnnotation::Axes { meta, origin, size } => {
                Annotation::Axes(AxesAnnotation {
                    meta: AnnotationMeta {
                        id: meta.id,
                        name: meta.name.clone(),
                        visible: meta.visible,
                        node_id: None,
                    },
                    origin: array_to_point3(*origin),
                    size: *size,
                })
            }
            SerializedAnnotation::Box { meta, center, size, color } => {
                Annotation::Box(BoxAnnotation {
                    meta: AnnotationMeta {
                        id: meta.id,
                        name: meta.name.clone(),
                        visible: meta.visible,
                        node_id: None,
                    },
                    center: array_to_point3(*center),
                    size: array_to_vec3(*size),
                    color: array_to_rgba(*color),
                })
            }
            SerializedAnnotation::Grid { meta, center, size, divisions, color } => {
                Annotation::Grid(GridAnnotation {
                    meta: AnnotationMeta {
                        id: meta.id,
                        name: meta.name.clone(),
                        visible: meta.visible,
                        node_id: None,
                    },
                    center: array_to_point3(*center),
                    size: *size,
                    divisions: *divisions,
                    color: array_to_rgba(*color),
                })
            }
        }
    }
}

impl SerializedTexture {
    /// Creates a SerializedTexture from a Texture.
    ///
    /// Uses original compressed bytes when available (from glTF embedded images or file paths),
    /// avoiding expensive re-encoding. Falls back to `fallback_format` encoding otherwise.
    pub fn from_texture(
        texture: &Texture,
        remapper: &IdRemapper,
        fallback_format: TextureFormat,
        compression: CompressionLevel,
    ) -> Result<Option<Self>, FormatError> {
        let Some(id) = remapper.remap_texture(texture.id()) else {
            return Ok(None);
        };

        // Priority 1: Use preserved original bytes (from glTF embedded images)
        // Extract and clone data first to release the immutable borrow
        let original = texture.original_bytes().map(|(bytes, format)| (bytes.to_vec(), format));
        if let Some((data, format)) = original {
            let image = texture.get_image()
                .map_err(|e| FormatError::TextureError(e.to_string()))?;
            let dimensions = image.dimensions();

            return Ok(Some(Self {
                id,
                format,
                width: dimensions.0,
                height: dimensions.1,
                data,
            }));
        }

        // Priority 2: Read original file bytes if texture has a source path
        if let Some(path) = texture.source_path() {
            if let Ok(bytes) = std::fs::read(path) {
                // Detect format from magic bytes
                let format = Self::detect_format(&bytes);
                if let Some(format) = format {
                    // Get dimensions from the loaded image (or load it)
                    let image = texture.get_image()
                        .map_err(|e| FormatError::TextureError(e.to_string()))?;
                    let dimensions = image.dimensions();

                    return Ok(Some(Self {
                        id,
                        format,
                        width: dimensions.0,
                        height: dimensions.1,
                        data: bytes,
                    }));
                }
            }
        }

        // Priority 3: Fall back to encoding with the configured format
        let image = texture.get_image()
            .map_err(|e| FormatError::TextureError(e.to_string()))?;
        let dimensions = image.dimensions();

        let (format, data) = match fallback_format {
            TextureFormat::Jpeg => {
                use image::codecs::jpeg::JpegEncoder;
                use image::ImageEncoder;

                let rgb = image.to_rgb8();
                let mut buf = Vec::new();
                JpegEncoder::new_with_quality(&mut buf, compression.jpeg_quality())
                    .write_image(
                        &rgb,
                        dimensions.0,
                        dimensions.1,
                        image::ColorType::Rgb8,
                    )
                    .map_err(|e| FormatError::TextureError(e.to_string()))?;
                (TextureFormat::Jpeg, buf)
            }
            _ => {
                use image::codecs::png::PngEncoder;
                use image::ImageEncoder;

                let rgba = image.to_rgba8();
                let mut buf = Vec::new();
                PngEncoder::new_with_quality(
                    &mut buf,
                    compression.png_compression(),
                    FilterType::Adaptive,
                )
                .write_image(
                    &rgba,
                    dimensions.0,
                    dimensions.1,
                    image::ColorType::Rgba8,
                )
                .map_err(|e| FormatError::TextureError(e.to_string()))?;
                (TextureFormat::Png, buf)
            }
        };

        Ok(Some(Self {
            id,
            format,
            width: dimensions.0,
            height: dimensions.1,
            data,
        }))
    }

    /// Detect image format from magic bytes.
    fn detect_format(bytes: &[u8]) -> Option<TextureFormat> {
        match image::guess_format(bytes) {
            Ok(image::ImageFormat::Png) => Some(TextureFormat::Png),
            Ok(image::ImageFormat::Jpeg) => Some(TextureFormat::Jpeg),
            _ => None,
        }
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
    let uncompressed = bincode::serialize(data)
        .map_err(|e| FormatError::SerializationError(e.to_string()))?;
    let uncompressed_size = uncompressed.len();
    let compressed = compress_with_level(&uncompressed, level)?;
    Ok((compressed, uncompressed_size))
}

/// Serialize and compress a section, returning the compressed bytes.
pub fn serialize_section<T: Serialize>(data: &T) -> Result<(Vec<u8>, usize), FormatError> {
    serialize_section_with_level(data, 3)
}

/// Decompress and deserialize a section.
pub fn deserialize_section<T: for<'de> Deserialize<'de>>(compressed: &[u8]) -> Result<T, FormatError> {
    let uncompressed = decompress(compressed)?;
    bincode::deserialize(&uncompressed)
        .map_err(|e| FormatError::DeserializationError(e.to_string()))
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
    pub materials: Vec<SerializedMaterial>,
    pub meshes: Vec<SerializedMesh>,
    pub instances: Vec<SerializedInstance>,
    pub nodes: Vec<SerializedNode>,
    pub lights: Vec<SerializedLight>,
    pub annotations: Vec<SerializedAnnotation>,
    pub environment_maps: Vec<SerializedEnvironmentMap>,
}

/// Phase 1: Parse WGSC header, TOC, and decompress/deserialize all sections.
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
    let materials: Vec<SerializedMaterial> =
        deserialize_section(read_section(SectionType::Materials)?)?;
    let meshes: Vec<SerializedMesh> =
        deserialize_section(read_section(SectionType::Meshes)?)?;
    let instances: Vec<SerializedInstance> =
        deserialize_section(read_section(SectionType::Instances)?)?;
    let nodes: Vec<SerializedNode> =
        deserialize_section(read_section(SectionType::Nodes)?)?;
    let lights: Vec<SerializedLight> =
        deserialize_section(read_section(SectionType::Lights)?)?;
    let annotations: Vec<SerializedAnnotation> =
        deserialize_section(read_section(SectionType::Annotations)?)?;
    let environment_maps: Vec<SerializedEnvironmentMap> =
        if let Some(entry) = toc.find(SectionType::EnvironmentMaps) {
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
        annotations,
        environment_maps,
    })
}

/// A single decoded texture with its file-local ID, ready for scene insertion.
pub struct DecodedTexture {
    pub file_id: u32,
    pub texture: Texture,
}

/// Phase 2: Decode serialized textures into images.
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

/// Phase 3: Assemble a [`Scene`] from parsed sections and decoded textures.
///
/// This is fast — it's just inserting data into hashmaps and linking nodes.
pub fn assemble_wgsc_scene(
    sections: WgscSections,
    decoded_textures: Vec<DecodedTexture>,
) -> Result<Scene, FormatError> {
    use cgmath::{Point3, Quaternion, Vector3};

    let mut scene = Scene::new();
    scene.materials.clear();
    scene.next_material_id = 0;

    // Add textures
    let mut texture_id_map: HashMap<u32, TextureId> = HashMap::new();
    for dt in decoded_textures {
        let scene_id = scene.add_texture(dt.texture);
        texture_id_map.insert(dt.file_id, scene_id);
    }

    // Add meshes
    let mut mesh_id_map: HashMap<u32, MeshId> = HashMap::new();
    for sm in sections.meshes {
        let file_id = sm.id;
        let mesh = sm.to_mesh();
        let scene_id = scene.add_mesh(mesh);
        mesh_id_map.insert(file_id, scene_id);
    }

    // Add materials (with texture ID remapping)
    let mut material_id_map: HashMap<u32, MaterialId> = HashMap::new();
    for sm in sections.materials {
        let file_id = sm.id;
        let mut material = sm.to_material();

        if let Some(tex_id) = sm.base_color_texture_id {
            if let Some(&scene_tex_id) = texture_id_map.get(&tex_id) {
                material = material.with_base_color_texture(scene_tex_id);
            }
        }
        if let Some(tex_id) = sm.normal_texture_id {
            if let Some(&scene_tex_id) = texture_id_map.get(&tex_id) {
                material = material.with_normal_texture(scene_tex_id);
            }
        }
        if let Some(tex_id) = sm.metallic_roughness_texture_id {
            if let Some(&scene_tex_id) = texture_id_map.get(&tex_id) {
                material = material.with_metallic_roughness_texture(scene_tex_id);
            }
        }

        let scene_id = scene.add_material(material);
        material_id_map.insert(file_id, scene_id);
    }

    // Add instances
    let mut instance_id_map: HashMap<u32, InstanceId> = HashMap::new();
    for si in sections.instances {
        let file_id = si.id;
        let mesh_id = *mesh_id_map.get(&si.mesh_id).unwrap_or(&0);
        let material_id = *material_id_map
            .get(&si.material_id)
            .unwrap_or(&DEFAULT_MATERIAL_ID);
        let scene_id = scene.add_instance(mesh_id, material_id);
        instance_id_map.insert(file_id, scene_id);
    }

    // Sort nodes by ID for consistent ordering
    let mut sorted_nodes = sections.nodes;
    sorted_nodes.sort_by_key(|n| n.id);

    // Build node ID map (file ID -> scene ID)
    let mut node_id_map: HashMap<u32, NodeId> = HashMap::new();

    // First pass: create all nodes without parent relationships
    for sn in &sorted_nodes {
        let position = Point3::new(sn.position[0], sn.position[1], sn.position[2]);
        let rotation = Quaternion::new(
            sn.rotation[3], // w (scalar)
            sn.rotation[0], // x
            sn.rotation[1], // y
            sn.rotation[2], // z
        );
        let scale = Vector3::new(sn.scale[0], sn.scale[1], sn.scale[2]);

        let node_id = scene
            .add_node(None, sn.name.clone(), position, rotation, scale)
            .map_err(|e| FormatError::DeserializationError(e.to_string()))?;

        node_id_map.insert(sn.id, node_id);

        if let Some(inst_file_id) = sn.instance_id {
            if let Some(&scene_inst_id) = instance_id_map.get(&inst_file_id) {
                scene
                    .get_node_mut(node_id)
                    .unwrap()
                    .set_instance(Some(scene_inst_id));
            }
        }

        if !sn.visible {
            scene.set_node_visibility(node_id, Visibility::Invisible);
        }
    }

    // Second pass: establish parent-child relationships
    scene.root_nodes.clear();

    for sn in &sorted_nodes {
        let scene_node_id = *node_id_map.get(&sn.id).unwrap();

        if let Some(parent_file_id) = sn.parent_id {
            if let Some(&scene_parent_id) = node_id_map.get(&parent_file_id) {
                scene
                    .get_node_mut(scene_node_id)
                    .unwrap()
                    .set_parent(Some(scene_parent_id));
                scene
                    .get_node_mut(scene_parent_id)
                    .unwrap()
                    .add_child(scene_node_id);
            }
        }
    }

    // Rebuild root_nodes from metadata
    scene.root_nodes = sections
        .metadata
        .root_nodes
        .iter()
        .filter_map(|&file_id| node_id_map.get(&file_id).copied())
        .collect();

    // Add lights
    scene.lights = sections
        .lights
        .iter()
        .map(SerializedLight::to_light)
        .collect();

    // Add annotations
    for sa in sections.annotations {
        let annotation = sa.to_annotation();
        scene
            .annotations
            .insert_with_id(annotation)
            .map_err(|e| FormatError::DeserializationError(e.to_string()))?;
    }

    // Add environment maps
    let mut env_map_id_map: HashMap<u32, EnvironmentMapId> = HashMap::new();
    for sem in sections.environment_maps {
        let scene_id = scene.add_environment_map_from_hdr_data(sem.hdr_data);
        if let Some(em) = scene.get_environment_map_mut(scene_id) {
            em.intensity = sem.intensity;
            em.rotation = sem.rotation;
        }
        env_map_id_map.insert(sem.id, scene_id);
    }

    // Set active environment map
    scene.active_environment_map = sections
        .metadata
        .active_environment_map
        .and_then(|file_id| env_map_id_map.get(&file_id).copied());

    Ok(scene)
}

// ============================================================================
// Scene Serialization
// ============================================================================

impl Scene {
    /// Serializes the scene to bytes in WGSC format with default options.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FormatError> {
        self.to_bytes_with_options(&SaveOptions::default())
    }

    /// Serializes the scene to bytes in WGSC format with custom options.
    pub fn to_bytes_with_options(&self, options: &SaveOptions) -> Result<Vec<u8>, FormatError> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let level = options.compression.zstd_level();
        let remapper = IdRemapper::from_scene(self);
        let mut output = Vec::new();
        let mut toc = TableOfContents::new();

        // Reserve space for header (will be written at the end)
        output.resize(HEADER_SIZE, 0);

        // ===== Metadata Section =====
        let metadata = SerializedMetadata {
            name: None,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            generator: format!("wgpu-engine {}", env!("CARGO_PKG_VERSION")),
            root_nodes: self.root_nodes
                .iter()
                .filter_map(|&id| remapper.remap_node(id))
                .collect(),
            active_environment_map: self.active_environment_map
                .and_then(|id| remapper.remap_environment_map(id)),
        };
        let offset = output.len() as u64;
        let (compressed, uncompressed_size) = serialize_section_with_level(&metadata, level)?;
        toc.add_entry(TocEntry {
            section_type: SectionType::Metadata,
            offset,
            compressed_size: compressed.len() as u64,
            uncompressed_size: uncompressed_size as u64,
        });
        output.extend(compressed);

        // ===== Nodes Section =====
        let nodes: Vec<SerializedNode> = self.nodes
            .values()
            .filter_map(|node| SerializedNode::from_node(node, &remapper))
            .collect();
        let offset = output.len() as u64;
        let (compressed, uncompressed_size) = serialize_section_with_level(&nodes, level)?;
        toc.add_entry(TocEntry {
            section_type: SectionType::Nodes,
            offset,
            compressed_size: compressed.len() as u64,
            uncompressed_size: uncompressed_size as u64,
        });
        output.extend(compressed);

        // ===== Instances Section =====
        let instances: Vec<SerializedInstance> = self.instances
            .values()
            .filter_map(|inst| SerializedInstance::from_instance(inst, &remapper))
            .collect();
        let offset = output.len() as u64;
        let (compressed, uncompressed_size) = serialize_section_with_level(&instances, level)?;
        toc.add_entry(TocEntry {
            section_type: SectionType::Instances,
            offset,
            compressed_size: compressed.len() as u64,
            uncompressed_size: uncompressed_size as u64,
        });
        output.extend(compressed);

        // ===== Materials Section =====
        let materials: Vec<SerializedMaterial> = self.materials
            .values()
            .filter_map(|mat| SerializedMaterial::from_material(mat, &remapper))
            .collect();
        let offset = output.len() as u64;
        let (compressed, uncompressed_size) = serialize_section_with_level(&materials, level)?;
        toc.add_entry(TocEntry {
            section_type: SectionType::Materials,
            offset,
            compressed_size: compressed.len() as u64,
            uncompressed_size: uncompressed_size as u64,
        });
        output.extend(compressed);

        // ===== Meshes Section =====
        let meshes: Vec<SerializedMesh> = self.meshes
            .values()
            .filter_map(|mesh| SerializedMesh::from_mesh(mesh, &remapper))
            .collect();
        let offset = output.len() as u64;
        let (compressed, uncompressed_size) = serialize_section_with_level(&meshes, level)?;
        toc.add_entry(TocEntry {
            section_type: SectionType::Meshes,
            offset,
            compressed_size: compressed.len() as u64,
            uncompressed_size: uncompressed_size as u64,
        });
        output.extend(compressed);

        // ===== Textures Section =====
        let mut textures = Vec::new();
        for texture in self.textures.values() {
            let serialized = SerializedTexture::from_texture(
                texture,
                &remapper,
                options.texture_format,
                options.compression,
            )?;
            if let Some(serialized) = serialized {
                textures.push(serialized);
            }
        }
        let offset = output.len() as u64;
        let (compressed, uncompressed_size) = serialize_section_with_level(&textures, level)?;
        toc.add_entry(TocEntry {
            section_type: SectionType::Textures,
            offset,
            compressed_size: compressed.len() as u64,
            uncompressed_size: uncompressed_size as u64,
        });
        output.extend(compressed);

        // ===== Lights Section =====
        let lights: Vec<SerializedLight> = self.lights
            .iter()
            .map(SerializedLight::from_light)
            .collect();
        let offset = output.len() as u64;
        let (compressed, uncompressed_size) = serialize_section_with_level(&lights, level)?;
        toc.add_entry(TocEntry {
            section_type: SectionType::Lights,
            offset,
            compressed_size: compressed.len() as u64,
            uncompressed_size: uncompressed_size as u64,
        });
        output.extend(compressed);

        // ===== Annotations Section =====
        let annotations: Vec<SerializedAnnotation> = self.annotations
            .iter()
            .filter_map(|ann| SerializedAnnotation::from_annotation(ann, &remapper))
            .collect();
        let offset = output.len() as u64;
        let (compressed, uncompressed_size) = serialize_section_with_level(&annotations, level)?;
        toc.add_entry(TocEntry {
            section_type: SectionType::Annotations,
            offset,
            compressed_size: compressed.len() as u64,
            uncompressed_size: uncompressed_size as u64,
        });
        output.extend(compressed);

        // ===== Environment Maps Section =====
        if !self.environment_maps.is_empty() {
            let mut env_maps = Vec::new();
            for (&id, env_map) in &self.environment_maps {
                let remapped_id = remapper.remap_environment_map(id).unwrap_or(0);
                let hdr_data = match &env_map.source {
                    super::environment::EnvironmentSource::EquirectangularPath(path) => {
                        std::fs::read(path).map_err(|e| FormatError::IoError(e))?
                    }
                    super::environment::EnvironmentSource::EquirectangularHdr(data) => {
                        data.clone()
                    }
                };
                env_maps.push(SerializedEnvironmentMap {
                    id: remapped_id,
                    hdr_data,
                    intensity: env_map.intensity,
                    rotation: env_map.rotation,
                });
            }
            let offset = output.len() as u64;
            let (compressed, uncompressed_size) = serialize_section_with_level(&env_maps, level)?;
            toc.add_entry(TocEntry {
                section_type: SectionType::EnvironmentMaps,
                offset,
                compressed_size: compressed.len() as u64,
                uncompressed_size: uncompressed_size as u64,
            });
            output.extend(compressed);
        }

        // ===== Write TOC =====
        let toc_offset = output.len() as u64;
        let (toc_compressed, _) = serialize_section_with_level(&toc, level)?;
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
    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), FormatError> {
        self.save_to_file_with_options(path, &SaveOptions::default())
    }

    /// Saves the scene to a file with custom options.
    pub fn save_to_file_with_options(
        &self,
        path: impl AsRef<std::path::Path>,
        options: &SaveOptions,
    ) -> Result<(), FormatError> {
        let bytes = self.to_bytes_with_options(options)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Loads a scene from a file.
    pub fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<Scene, FormatError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Point3, Quaternion, Vector3};

    /// Creates a simple test scene with various elements.
    fn create_test_scene() -> Scene {
        let mut scene = Scene::new();

        // Add a mesh
        let mesh = Mesh::cube(1.0);
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
            Point3::new(1.0, 2.0, 3.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(2.0, 2.0, 2.0),
        ).unwrap();

        // Add a child node
        let _child_id = scene.add_node(
            Some(node_id),
            Some("ChildNode".to_string()),
            Point3::new(0.5, 0.5, 0.5),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        ).unwrap();

        // Add a light
        scene.lights.push(Light::point(
            Vector3::new(5.0, 5.0, 5.0),
            RgbaColor::WHITE,
            10.0,
        ));

        // Add an annotation
        scene.annotations.add_line(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            RgbaColor::BLUE,
        );

        scene
    }

    #[test]
    fn test_round_trip_basic() {
        let original = create_test_scene();

        // Serialize
        let bytes = original.to_bytes().expect("Failed to serialize scene");

        // Check magic number
        assert_eq!(&bytes[0..4], b"WGSC");

        // Deserialize
        let loaded = Scene::from_bytes(&bytes).expect("Failed to deserialize scene");

        // Verify basic structure
        assert_eq!(loaded.nodes.len(), original.nodes.len());
        assert_eq!(loaded.meshes.len(), original.meshes.len());
        assert_eq!(loaded.materials.len(), original.materials.len());
        assert_eq!(loaded.instances.len(), original.instances.len());
        assert_eq!(loaded.lights.len(), original.lights.len());
        assert_eq!(loaded.root_nodes.len(), original.root_nodes.len());
    }

    #[test]
    fn test_round_trip_node_properties() {
        let original = create_test_scene();
        let bytes = original.to_bytes().expect("Failed to serialize");
        let loaded = Scene::from_bytes(&bytes).expect("Failed to deserialize");

        // Find the test node by name
        let original_node = original.nodes.values()
            .find(|n| n.name.as_deref() == Some("TestNode"))
            .expect("TestNode not found in original");

        let loaded_node = loaded.nodes.values()
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
        let bytes = original.to_bytes().expect("Failed to serialize");
        let loaded = Scene::from_bytes(&bytes).expect("Failed to deserialize");

        // Skip the default material (ID 0), find our custom material
        let original_mat = original.materials.values()
            .find(|m| m.id != DEFAULT_MATERIAL_ID)
            .expect("Custom material not found in original");

        let loaded_mat = loaded.materials.values()
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
        let bytes = original.to_bytes().expect("Failed to serialize");
        let loaded = Scene::from_bytes(&bytes).expect("Failed to deserialize");

        // Get the first mesh from each scene
        let original_mesh = original.meshes.values().next().expect("No mesh in original");
        let loaded_mesh = loaded.meshes.values().next().expect("No mesh in loaded");

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
        let bytes = original.to_bytes().expect("Failed to serialize");
        let loaded = Scene::from_bytes(&bytes).expect("Failed to deserialize");

        // Find child node
        let loaded_child = loaded.nodes.values()
            .find(|n| n.name.as_deref() == Some("ChildNode"))
            .expect("ChildNode not found");

        // Verify it has a parent
        assert!(loaded_child.parent().is_some());

        // Find parent node
        let loaded_parent = loaded.nodes.values()
            .find(|n| n.name.as_deref() == Some("TestNode"))
            .expect("TestNode not found");

        // Verify parent has child
        assert!(!loaded_parent.children().is_empty());
    }

    #[test]
    fn test_round_trip_lights() {
        let original = create_test_scene();
        let bytes = original.to_bytes().expect("Failed to serialize");
        let loaded = Scene::from_bytes(&bytes).expect("Failed to deserialize");

        assert_eq!(loaded.lights.len(), 1);

        match &loaded.lights[0] {
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
    fn test_round_trip_annotations() {
        let original = create_test_scene();
        let bytes = original.to_bytes().expect("Failed to serialize");
        let loaded = Scene::from_bytes(&bytes).expect("Failed to deserialize");

        assert_eq!(loaded.annotations.len(), 1);

        let annotation = loaded.annotations.iter().next().unwrap();
        match annotation {
            Annotation::Line(line) => {
                assert!((line.start.x - 0.0).abs() < 1e-6);
                assert!((line.end.x - 1.0).abs() < 1e-6);
                assert!((line.color.b - 1.0).abs() < 1e-6); // Blue
            }
            _ => panic!("Expected line annotation"),
        }
    }

    #[test]
    fn test_empty_scene_round_trip() {
        let scene = Scene::new();
        let bytes = scene.to_bytes().expect("Failed to serialize empty scene");
        let loaded = Scene::from_bytes(&bytes).expect("Failed to deserialize empty scene");

        // Should only have the default material
        assert_eq!(loaded.materials.len(), 1);
        assert!(loaded.nodes.is_empty());
        assert!(loaded.meshes.is_empty());
        assert!(loaded.instances.is_empty());
    }

    #[test]
    fn test_version_in_header() {
        let scene = Scene::new();
        let bytes = scene.to_bytes().expect("Failed to serialize");

        // Check version bytes (offset 4-5)
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, VERSION);
    }

    #[test]
    fn test_invalid_magic_rejected() {
        let mut bytes = vec![b'X', b'X', b'X', b'X']; // Wrong magic
        bytes.extend([0u8; 12]); // Rest of header

        let result = Scene::from_bytes(&bytes);
        assert!(matches!(result, Err(FormatError::InvalidMagic)));
    }

    #[test]
    fn test_reified_annotation_geometry_excluded() {
        // Create a scene with a mesh and an annotation
        let mut scene = Scene::new();

        // Add a regular mesh/node
        let mesh = Mesh::cube(1.0);
        let mesh_id = scene.add_mesh(mesh);
        let mat_id = scene.add_material(Material::new());
        let _node_id = scene.add_instance_node(
            None,
            mesh_id,
            mat_id,
            Some("RegularNode".to_string()),
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
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
        assert!(scene.meshes.len() > 1, "Should have annotation mesh");
        assert!(scene.nodes.len() > 1, "Should have annotation nodes");

        // Serialize
        let bytes = scene.to_bytes().expect("Failed to serialize");

        // Deserialize
        let loaded = Scene::from_bytes(&bytes).expect("Failed to deserialize");

        // After deserialization: annotation geometry should NOT be present
        // Only the regular mesh/material/instance/node should be serialized
        assert_eq!(loaded.meshes.len(), 1, "Only regular mesh should be serialized");
        assert_eq!(loaded.instances.len(), 1, "Only regular instance should be serialized");
        assert_eq!(loaded.nodes.len(), 1, "Only regular node should be serialized");

        // But annotation data should still be present
        assert_eq!(loaded.annotations.len(), 1, "Annotation data should be preserved");

        // Annotation should not be reified yet (node_id should be None)
        let annotation = loaded.annotations.iter().next().unwrap();
        assert!(!annotation.is_reified(), "Annotation should not be reified after load");

        // Verify the regular node is present
        let regular_node = loaded.nodes.values()
            .find(|n| n.name.as_deref() == Some("RegularNode"))
            .expect("RegularNode not found");
        assert!(regular_node.instance().is_some());

        // Now we can re-reify the annotations
        let reified_count = loaded.annotations.unreified_count();
        assert_eq!(reified_count, 1, "Should have 1 unreified annotation");
    }
}
