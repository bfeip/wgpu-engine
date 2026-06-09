//! Scene file format serialization.
//!
//! This module provides serialization and deserialization of scenes to a custom
//! binary format (.duck). The format is designed to be:
//! - Concise: Uses Zstd compression for most resource types
//! - Random-accessible: Per-resource TOC with stable UUIDs
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
//! │ RESOURCE DATA (variable, one blob per resource)              │
//! ├──────────────────────────────────────────────────────────────┤
//! │ TABLE OF CONTENTS (zstd-compressed bincode Vec<TocEntry>)    │
//! └──────────────────────────────────────────────────────────────┘
//! ```

pub mod wire;
pub use wire::*;

use std::collections::HashSet;
use std::io::Cursor;

use image::codecs::png::CompressionType;
use serde::Serialize;
use serde::de::DeserializeOwned;
use thiserror::Error;

use duck_engine_scene::{
    FaceMaterial,
    FaceMaterialId,
    GenericId,
    Instance,
    InstanceId,
    LineMaterial,
    LineMaterialId,
    Mesh,
    MeshId,
    Node,
    NodeFlags,
    NodeId,
    NodePayload,
    PointMaterial,
    PointMaterialId,
    Texture,
    TextureId,
    EnvironmentMap, Scene,
};

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

    #[error("Missing required metadata resource")]
    MissingMetadata,

    #[error("Texture load error: {0}")]
    TextureError(String),
}

/// Controls the compression/quality tradeoff for saving scenes.
///
/// Affects zstd data compression and PNG compression.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompressionLevel {
    /// Fastest compression, larger files. Zstd level 1, PNG fast.
    Fast,
    /// Balanced compression (default). Zstd level 3, PNG default.
    #[default]
    Default,
    /// Best compression, smaller files. Zstd level 19, PNG best.
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

/// Compress data using Zstd with the specified compression level.
fn compress(data: &[u8], level: i32) -> Result<Vec<u8>, FormatError> {
    zstd::encode_all(Cursor::new(data), level)
        .map_err(|e| FormatError::CompressionError(e.to_string()))
}

/// Decompress Zstd-compressed data.
fn decompress(data: &[u8]) -> Result<Vec<u8>, FormatError> {
    zstd::decode_all(Cursor::new(data))
        .map_err(|e| FormatError::DecompressionError(e.to_string()))
}

/// Bincode-encode then zstd-compress a value. Used for all non-texture resources.
fn encode_resource<T: Serialize>(data: &T, level: i32) -> Result<Vec<u8>, FormatError> {
    let uncompressed = bincode::serde::encode_to_vec(data, bincode::config::standard())
        .map_err(|e| FormatError::SerializationError(e.to_string()))?;
    compress(&uncompressed, level)
}

/// Zstd-decompress then bincode-decode a resource. Useful when reading raw resource bytes
/// directly from a WGSC file without going through [`from_bytes`].
pub fn decode_resource<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, FormatError> {
    let uncompressed = decompress(bytes)?;
    let (value, _) = bincode::serde::decode_from_slice(&uncompressed, bincode::config::standard())
        .map_err(|e| FormatError::DeserializationError(e.to_string()))?;
    Ok(value)
}

/// Append pre-encoded resource bytes to the output buffer and record the TOC entry.
fn append_resource(
    resource_bytes: Vec<u8>,
    resource_type: ResourceType,
    resource_id: GenericId,
    output: &mut Vec<u8>,
    toc: &mut Vec<TocEntry>,
) {
    let offset = output.len() as u64;
    let size = resource_bytes.len() as u32;
    toc.push(TocEntry { resource_type, resource_id, offset, size });
    output.extend(resource_bytes);
}

/// All deserialized resources from a Duck file, ready for scene assembly.
/// Produced by [`parse_scene`], consumed by [`assemble_scene`].
struct ParsedSceneData {
    metadata: SerializedMetadata,
    textures: Vec<Texture>,
    face_materials: Vec<FaceMaterial>,
    line_materials: Vec<LineMaterial>,
    point_materials: Vec<PointMaterial>,
    meshes: Vec<Mesh>,
    instances: Vec<Instance>,
    nodes: Vec<Node>,
    environment_maps: Vec<EnvironmentMap>,
}

/// Parse Duck header, TOC, and decompress/deserialize all resources.
fn parse_scene(bytes: &[u8]) -> Result<ParsedSceneData, FormatError> {
    let mut cursor = Cursor::new(bytes);
    let header = FileHeader::read(&mut cursor)?;

    let toc_data = &bytes[header.toc_offset as usize..];
    let toc: Vec<TocEntry> = decode_resource(toc_data)?;

    let resource_bytes = |entry: &TocEntry| -> &[u8] {
        let start = entry.offset as usize;
        let end = start + entry.size as usize;
        &bytes[start..end]
    };

    let mut metadata: Option<SerializedMetadata> = None;
    let mut textures: Vec<Texture> = Vec::new();
    let mut face_materials: Vec<FaceMaterial> = Vec::new();
    let mut line_materials: Vec<LineMaterial> = Vec::new();
    let mut point_materials: Vec<PointMaterial> = Vec::new();
    let mut meshes: Vec<Mesh> = Vec::new();
    let mut instances: Vec<Instance> = Vec::new();
    let mut nodes: Vec<Node> = Vec::new();
    let mut environment_maps: Vec<EnvironmentMap> = Vec::new();

    for entry in &toc {
        match entry.resource_type {
            ResourceType::Metadata => {
                metadata = Some(decode_resource(resource_bytes(entry))?);
            }
            ResourceType::Node => {
                nodes.push(decode_resource(resource_bytes(entry))?);
            }
            ResourceType::Instance => {
                instances.push(decode_resource(resource_bytes(entry))?);
            }
            ResourceType::FaceMaterial => {
                face_materials.push(decode_resource(resource_bytes(entry))?);
            }
            ResourceType::LineMaterial => {
                line_materials.push(decode_resource(resource_bytes(entry))?);
            }
            ResourceType::PointMaterial => {
                point_materials.push(decode_resource(resource_bytes(entry))?);
            }
            ResourceType::Mesh => {
                meshes.push(decode_resource(resource_bytes(entry))?);
            }
            ResourceType::Texture => {
                let raw = resource_bytes(entry).to_vec();
                let tex = Texture::from_image_bytes_with_id(entry.resource_id.cast(), raw)
                    .map_err(|e| FormatError::TextureError(e.to_string()))?;
                textures.push(tex);
            }
            ResourceType::EnvironmentMap => {
                environment_maps.push(decode_resource(resource_bytes(entry))?);
            }
        }
    }

    let metadata = metadata.ok_or(FormatError::MissingMetadata)?;

    Ok(ParsedSceneData {
        metadata,
        textures,
        face_materials,
        line_materials,
        point_materials,
        meshes,
        instances,
        nodes,
        environment_maps,
    })
}

/// Assemble a [`Scene`] from parsed sections.
fn assemble_scene(sections: ParsedSceneData) -> Result<Scene, FormatError> {
    let mut scene = Scene::new();

    for texture in sections.textures {
        scene.add_texture(texture);
    }

    for mesh in sections.meshes {
        scene.add_mesh(mesh);
    }

    for material in sections.face_materials {
        scene.add_face_material(material);
    }

    for material in sections.line_materials {
        scene.add_line_material(material);
    }

    for material in sections.point_materials {
        scene.add_point_material(material);
    }

    for instance in sections.instances {
        scene.add_instance(instance);
    }

    for node in sections.nodes {
        scene.insert_node(node);
    }

    for em in sections.environment_maps {
        scene.add_environment_map(em);
    }

    scene.set_active_environment_map(sections.metadata.active_environment_map);

    Ok(scene)
}

/// Estimates the serialized size in bytes for buffer pre-allocation.
///
/// Returns a heuristic upper bound — intentionally overestimates to avoid
/// reallocations. The actual size after zstd compression will typically be
/// smaller.
pub fn estimate_serialized_size(scene: &Scene) -> usize {
    use duck_engine_scene::EnvironmentSource;

    const GOOD_COMPRESSION_RATIO: f64 = 0.6;

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

    let texture_estimate: usize = scene.textures()
        .map(|tex| {
            if let Some((bytes, _)) = tex.original_bytes() {
                return bytes.len();
            }
            if let Some(path) = tex.source_path()
                && let Ok(meta) = std::fs::metadata(path) {
                    return meta.len() as usize;
                }
            tex.dimensions()
                .map(|(w, h)| {
                    let dimensions = w as usize * h as usize;
                    let uncompressed_size = dimensions * 4;
                    (uncompressed_size as f64 * GOOD_COMPRESSION_RATIO) as usize
                })
                .unwrap_or(0)
        })
        .sum();

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

    let material_count =
        scene.face_material_count() + scene.line_material_count() + scene.point_material_count();
    let structured = scene.node_count() * std::mem::size_of::<Node>()
        + scene.instance_count() * std::mem::size_of::<Instance>()
        + scene.face_material_count() * std::mem::size_of::<FaceMaterial>()
        + scene.line_material_count() * std::mem::size_of::<LineMaterial>()
        + scene.point_material_count() * std::mem::size_of::<PointMaterial>();
    let structured_estimate = (structured as f64 * GOOD_COMPRESSION_RATIO) as usize;

    // 256-byte base covers metadata and zstd framing overhead on small scenes.
    // 64 bytes per resource accounts for zstd per-frame overhead on small payloads.
    let n_resources = scene.node_count()
        + scene.instance_count()
        + material_count
        + scene.mesh_count()
        + scene.texture_count()
        + scene.environment_map_count()
        + 1; // metadata
    let overhead = HEADER_SIZE + n_resources * 64 + 256;

    let total = mesh_estimate + texture_estimate + env_estimate + structured_estimate + overhead;

    total + total / 10
}

struct ExportableResources {
    root_nodes: Vec<NodeId>,
    nodes: HashSet<NodeId>,
    instances: HashSet<InstanceId>,
    meshes: HashSet<MeshId>,
    face_materials: HashSet<FaceMaterialId>,
    line_materials: HashSet<LineMaterialId>,
    point_materials: HashSet<PointMaterialId>,
    textures: HashSet<TextureId>,
}

/// Collects the IDs of all resources that should appear in an export.
///
/// Walks the scene tree, skipping subtrees rooted at a `DO_NOT_EXPORT` node, then
/// follows instance → mesh/material → texture references to determine the complete
/// set of reachable resources.
fn collect_exportable(scene: &Scene) -> ExportableResources {
    fn visit(
        node_id: NodeId,
        scene: &Scene,
        nodes: &mut HashSet<NodeId>,
        instances: &mut HashSet<InstanceId>,
    ) {
        let Some(node) = scene.get_node(node_id) else { return };
        if node.flags().contains(NodeFlags::DO_NOT_EXPORT) {
            return;
        }
        nodes.insert(node_id);
        if let NodePayload::Instance(iid) = node.payload() {
            instances.insert(*iid);
        }
        for &child_id in node.children() {
            visit(child_id, scene, nodes, instances);
        }
    }

    let mut res = ExportableResources {
        root_nodes: Vec::new(),
        nodes: HashSet::new(),
        instances: HashSet::new(),
        meshes: HashSet::new(),
        face_materials: HashSet::new(),
        line_materials: HashSet::new(),
        point_materials: HashSet::new(),
        textures: HashSet::new(),
    };

    for &root_id in scene.root_nodes() {
        let is_exportable = scene
            .get_node(root_id)
            .map_or(false, |n| !n.flags().contains(NodeFlags::DO_NOT_EXPORT));
        if is_exportable {
            res.root_nodes.push(root_id);
        }
        visit(root_id, scene, &mut res.nodes, &mut res.instances);
    }

    for &iid in &res.instances {
        let Some(instance) = scene.get_instance(iid) else { continue };
        res.meshes.insert(instance.mesh());
        if let Some(id) = instance.face_material() {
            res.face_materials.insert(id);
        }
        if let Some(id) = instance.line_material() {
            res.line_materials.insert(id);
        }
        if let Some(id) = instance.point_material() {
            res.point_materials.insert(id);
        }
    }

    for &mid in &res.face_materials {
        let Some(mat) = scene.get_face_material(mid) else { continue };
        for tex_id in [mat.base_color_texture(), mat.normal_texture(), mat.metallic_roughness_texture()].into_iter().flatten() {
            res.textures.insert(tex_id);
        }
    }

    res
}

/// Serializes the scene to bytes in Duck format with default options.
pub fn to_bytes(scene: &Scene) -> Result<Vec<u8>, FormatError> {
    to_bytes_with_options(scene, &SaveOptions::default())
}

/// Serializes the scene to bytes in Duck format with custom options.
pub fn to_bytes_with_options(scene: &Scene, options: &SaveOptions) -> Result<Vec<u8>, FormatError> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let exportable = collect_exportable(scene);
    let compression_level = options.compression.zstd_level();
    let mut output = Vec::with_capacity(estimate_serialized_size(scene));
    output.resize(HEADER_SIZE, 0);
    let mut toc: Vec<TocEntry> = Vec::new();

    let metadata = SerializedMetadata {
        name: None,
        created_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        generator: format!("duck-engine {}", env!("CARGO_PKG_VERSION")),
        root_nodes: exportable.root_nodes,
        active_environment_map: scene.active_environment_map(),
    };
    append_resource(
        encode_resource(&metadata, compression_level)?,
        ResourceType::Metadata,
        GenericId::nil(),
        &mut output,
        &mut toc,
    );

    for &node_id in &exportable.nodes {
        let Some(node) = scene.get_node(node_id) else { continue };
        append_resource(
            encode_resource(node, compression_level)?,
            ResourceType::Node,
            node.id.erased(),
            &mut output,
            &mut toc,
        );
    }

    for &instance_id in &exportable.instances {
        let Some(instance) = scene.get_instance(instance_id) else { continue };
        append_resource(
            encode_resource(instance, compression_level)?,
            ResourceType::Instance,
            instance.id.erased(),
            &mut output,
            &mut toc,
        );
    }

    for &material_id in &exportable.face_materials {
        let Some(material) = scene.get_face_material(material_id) else { continue };
        append_resource(
            encode_resource(material, compression_level)?,
            ResourceType::FaceMaterial,
            material.id.erased(),
            &mut output,
            &mut toc,
        );
    }

    for &material_id in &exportable.line_materials {
        let Some(material) = scene.get_line_material(material_id) else { continue };
        append_resource(
            encode_resource(material, compression_level)?,
            ResourceType::LineMaterial,
            material.id.erased(),
            &mut output,
            &mut toc,
        );
    }

    for &material_id in &exportable.point_materials {
        let Some(material) = scene.get_point_material(material_id) else { continue };
        append_resource(
            encode_resource(material, compression_level)?,
            ResourceType::PointMaterial,
            material.id.erased(),
            &mut output,
            &mut toc,
        );
    }

    for &mesh_id in &exportable.meshes {
        let Some(mesh) = scene.get_mesh(mesh_id) else { continue };
        append_resource(
            encode_resource(mesh, compression_level)?,
            ResourceType::Mesh,
            mesh.id.erased(),
            &mut output,
            &mut toc,
        );
    }

    for &tex_id in &exportable.textures {
        let Some(tex) = scene.get_texture(tex_id) else { continue };
        let image_bytes = encode_texture(tex, options)?;
        append_resource(
            image_bytes,
            ResourceType::Texture,
            tex.id.erased(),
            &mut output,
            &mut toc,
        );
    }

    for em in scene.environment_maps() {
        append_resource(
            encode_resource(em, compression_level)?,
            ResourceType::EnvironmentMap,
            em.id.erased(),
            &mut output,
            &mut toc,
        );
    }

    let toc_offset = output.len() as u64;
    let toc_bytes = encode_resource(&toc, compression_level)?;
    output.extend(toc_bytes);

    let header = FileHeader::new(toc_offset);
    let mut header_bytes = Vec::new();
    header.write(&mut header_bytes)?;
    output[..HEADER_SIZE].copy_from_slice(&header_bytes);

    Ok(output)
}

/// Encode a texture to its raw PNG/JPEG bytes for embedding in the file.
fn encode_texture(tex: &Texture, options: &SaveOptions) -> Result<Vec<u8>, FormatError> {
    if let Some((bytes, _)) = tex.original_bytes() {
        return Ok(bytes.to_vec());
    }
    if let Some(path) = tex.source_path() {
        return std::fs::read(path).map_err(FormatError::IoError);
    }
    let image = tex.get_image()
        .map_err(|e| FormatError::TextureError(e.to_string()))?;
    let mut buf = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
        .map_err(|e| FormatError::TextureError(e.to_string()))?;
    let _ = options.compression.png_compression();
    Ok(buf)
}

/// Deserializes a scene from Duck format bytes.
pub fn from_bytes(bytes: &[u8]) -> Result<Scene, FormatError> {
    assemble_scene(parse_scene(bytes)?)
}

/// Saves the scene to a file.
pub fn save_to_file(
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

#[cfg(test)]
mod tests;
