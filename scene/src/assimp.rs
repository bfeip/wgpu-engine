//! Assimp-based scene loading.
//!
//! Uses the russimp crate (Rust bindings for the Open Asset Import Library) to load
//! 3D models in dozens of formats (FBX, DAE, 3DS, Blend, STL, PLY, OBJ, etc.)
//! and convert them into the engine's Scene representation.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use anyhow::{Result, anyhow};
use cgmath::{Matrix4, Point3, Quaternion, Vector3};
use russimp::material::{Material as RMaterial, TextureType};
use russimp::node::Node as RNode;
use russimp::scene::{PostProcess, Scene as RScene};

use crate::camera::Camera;
use crate::common::{RgbaColor, decompose_matrix};
use crate::{
    DEFAULT_MATERIAL_ID, Light, Material, MaterialId, Mesh, MeshId, MeshIndex, MeshPrimitive,
    NodeId, PrimitiveType, Scene, Texture, TextureId, Vertex,
};

/// Result of loading a scene via assimp.
pub struct AssimpLoadResult {
    pub scene: Scene,
    pub camera: Option<Camera>,
}

/// Default post-processing flags applied to every assimp load.
fn default_post_process() -> Vec<PostProcess> {
    vec![
        PostProcess::Triangulate,
        PostProcess::GenerateSmoothNormals,
        PostProcess::JoinIdenticalVertices,
        PostProcess::FlipUVs,
    ]
}

/// Load a scene from a file path using assimp.
pub fn load_assimp_scene_from_path(path: &Path) -> Result<AssimpLoadResult> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow!("Path contains non-UTF8 characters"))?;

    let assimp_scene = RScene::from_file(path_str, default_post_process())
        .map_err(|e| anyhow!("Assimp import failed: {:?}", e))?;

    let base_path = path.parent();
    convert_scene(&assimp_scene, base_path)
}

/// Load a scene from bytes in memory using assimp.
///
/// `hint` is a format hint like "fbx" or "dae" (without the dot).
pub fn load_assimp_scene_from_bytes(bytes: &[u8], hint: &str) -> Result<AssimpLoadResult> {
    let assimp_scene = RScene::from_buffer(bytes, default_post_process(), hint)
        .map_err(|e| anyhow!("Assimp import failed: {:?}", e))?;

    convert_scene(&assimp_scene, None)
}

// ============================================================================
// Conversion Pipeline
// ============================================================================

/// Convert a russimp Scene into our engine's Scene.
fn convert_scene(assimp_scene: &RScene, base_path: Option<&Path>) -> Result<AssimpLoadResult> {
    let mut scene = Scene::new();

    // Phase 1: Load textures (embedded + external referenced by materials)
    let texture_map = load_textures(assimp_scene, base_path, &mut scene);

    // Phase 2: Load materials
    let material_map = load_materials(&assimp_scene.materials, &texture_map, base_path, &mut scene);

    // Phase 3: Load meshes
    let mesh_map = load_meshes(&assimp_scene.meshes, &material_map, &mut scene);

    // Phase 4: Build node hierarchy
    if let Some(ref root) = assimp_scene.root {
        build_node_tree(root, &mut scene, &mesh_map, &material_map, None)?;
    }

    // Phase 5: Load lights
    load_lights(&assimp_scene.lights, &mut scene);

    // Phase 6: Set default lights if none were loaded
    scene.set_default_lights();

    // Phase 7: Extract camera
    let camera = extract_camera(&assimp_scene.cameras);

    Ok(AssimpLoadResult { scene, camera })
}

// ============================================================================
// Textures
// ============================================================================

/// Maps texture path/key → scene TextureId.
type TextureMap = HashMap<String, TextureId>;

/// Load embedded textures from assimp materials and build a path→TextureId map.
///
/// In russimp, embedded textures are accessed through material texture references.
/// We load them lazily when processing materials.
fn load_textures(
    assimp_scene: &RScene,
    _base_path: Option<&Path>,
    _scene: &mut Scene,
) -> TextureMap {
    let mut map = TextureMap::new();

    // Pre-load embedded textures from materials
    for mat in &assimp_scene.materials {
        for (_tex_type, tex_rc) in &mat.textures {
            let tex = tex_rc.borrow();
            let key = tex.filename.clone();
            if key.is_empty() || map.contains_key(&key) {
                continue;
            }

            // Try to decode embedded texture data
            match &tex.data {
                russimp::material::DataContent::Bytes(bytes) if !bytes.is_empty() => {
                    match image::load_from_memory(bytes) {
                        Ok(img) => {
                            let tex_id = _scene.add_texture(Texture::from_image(img));
                            map.insert(key, tex_id);
                        }
                        Err(e) => {
                            log::warn!("Failed to decode embedded texture '{}': {}", key, e);
                        }
                    }
                }
                russimp::material::DataContent::Texel(texels) if !texels.is_empty() => {
                    let width = tex.width;
                    let height = if tex.height == 0 { 1 } else { tex.height };
                    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
                    for texel in texels {
                        rgba.push(texel.r);
                        rgba.push(texel.g);
                        rgba.push(texel.b);
                        rgba.push(texel.a);
                    }
                    if let Some(img) = image::RgbaImage::from_raw(width, height, rgba) {
                        let tex_id =
                            _scene.add_texture(Texture::from_image(image::DynamicImage::ImageRgba8(img)));
                        map.insert(key, tex_id);
                    }
                }
                _ => {
                    // No embedded data — will be resolved as external file later
                }
            }
        }
    }

    map
}

/// Resolve a texture path from an assimp material, loading it if needed.
fn resolve_texture(
    path: &str,
    base_path: Option<&Path>,
    texture_map: &mut TextureMap,
    scene: &mut Scene,
) -> Option<TextureId> {
    // Check if already loaded (embedded textures use "*N" keys)
    if let Some(&id) = texture_map.get(path) {
        return Some(id);
    }

    // Try to resolve as external file
    let base = base_path?;
    let full_path = base.join(path);
    if !full_path.exists() {
        log::warn!("Texture file not found: {}", full_path.display());
        return None;
    }

    let tex_id = scene.add_texture(Texture::from_path(full_path));
    texture_map.insert(path.to_string(), tex_id);
    Some(tex_id)
}

// ============================================================================
// Materials
// ============================================================================

/// Load all assimp materials, returning a map from assimp material index → scene MaterialId.
fn load_materials(
    assimp_materials: &[RMaterial],
    texture_map: &TextureMap,
    base_path: Option<&Path>,
    scene: &mut Scene,
) -> Vec<MaterialId> {
    // We need a mutable texture map for lazy-loading external textures.
    // Clone the initial map so we can extend it.
    let mut tex_map = texture_map.clone();

    assimp_materials
        .iter()
        .map(|mat| load_single_material(mat, &mut tex_map, base_path, scene))
        .collect()
}

fn load_single_material(
    mat: &RMaterial,
    texture_map: &mut TextureMap,
    base_path: Option<&Path>,
    scene: &mut Scene,
) -> MaterialId {
    let mut material = Material::new();

    // Extract base color / diffuse color from properties
    if let Some(color) = extract_color_property(mat, "$clr.diffuse") {
        material = material.with_base_color_factor(color);
    } else if let Some(color) = extract_color_property(mat, "$clr.base") {
        material = material.with_base_color_factor(color);
    }

    // Extract metallic factor
    if let Some(val) = extract_float_property(mat, "$mat.metallicFactor") {
        material = material.with_metallic_factor(val);
    }

    // Extract roughness factor
    if let Some(val) = extract_float_property(mat, "$mat.roughnessFactor") {
        material = material.with_roughness_factor(val);
    }

    // Extract textures
    if let Some(tex_id) = extract_texture(mat, TextureType::Diffuse, texture_map, base_path, scene)
    {
        material = material.with_base_color_texture(tex_id);
    } else if let Some(tex_id) =
        extract_texture(mat, TextureType::BaseColor, texture_map, base_path, scene)
    {
        material = material.with_base_color_texture(tex_id);
    }

    if let Some(tex_id) =
        extract_texture(mat, TextureType::Normals, texture_map, base_path, scene)
    {
        material = material.with_normal_texture(tex_id);
    }

    if let Some(tex_id) =
        extract_texture(mat, TextureType::Metalness, texture_map, base_path, scene)
    {
        material = material.with_metallic_roughness_texture(tex_id);
    }

    // Set line/point colors to match base color for wireframe rendering
    let base_color = material.base_color_factor();
    material = material
        .with_line_color(base_color)
        .with_point_color(base_color);

    scene.add_material(material)
}

/// Extract a color property from an assimp material by key.
fn extract_color_property(mat: &RMaterial, key: &str) -> Option<RgbaColor> {
    for prop in &mat.properties {
        if prop.key == key {
            if let russimp::material::PropertyTypeInfo::FloatArray(ref floats) = prop.data {
                if floats.len() >= 3 {
                    let a = if floats.len() >= 4 { floats[3] } else { 1.0 };
                    return Some(RgbaColor { r: floats[0], g: floats[1], b: floats[2], a });
                }
            }
        }
    }
    None
}

/// Extract a float property from an assimp material by key.
fn extract_float_property(mat: &RMaterial, key: &str) -> Option<f32> {
    for prop in &mat.properties {
        if prop.key == key {
            if let russimp::material::PropertyTypeInfo::FloatArray(ref floats) = prop.data {
                if !floats.is_empty() {
                    return Some(floats[0]);
                }
            }
        }
    }
    None
}

/// Extract a texture from an assimp material for a given texture type.
fn extract_texture(
    mat: &RMaterial,
    tex_type: TextureType,
    texture_map: &mut TextureMap,
    base_path: Option<&Path>,
    scene: &mut Scene,
) -> Option<TextureId> {
    let tex = mat.textures.get(&tex_type)?;
    let tex_ref = tex.borrow();
    let path = &tex_ref.filename;
    if path.is_empty() {
        return None;
    }
    resolve_texture(path, base_path, texture_map, scene)
}

// ============================================================================
// Meshes
// ============================================================================

/// Loaded mesh info: Vec of (MeshId, MaterialId) pairs.
/// A single assimp mesh may produce multiple engine meshes if it exceeds the u16 index limit.
type MeshEntry = Vec<(MeshId, MaterialId)>;

/// Load all assimp meshes, returning a map from assimp mesh index → loaded mesh entries.
fn load_meshes(
    assimp_meshes: &[russimp::mesh::Mesh],
    material_map: &[MaterialId],
    scene: &mut Scene,
) -> Vec<MeshEntry> {
    assimp_meshes
        .iter()
        .map(|m| load_single_mesh(m, material_map, scene))
        .collect()
}

fn load_single_mesh(
    assimp_mesh: &russimp::mesh::Mesh,
    material_map: &[MaterialId],
    scene: &mut Scene,
) -> MeshEntry {
    let material_id = material_map
        .get(assimp_mesh.material_index as usize)
        .copied()
        .unwrap_or(DEFAULT_MATERIAL_ID);

    // Build vertex data
    let vertices: Vec<Vertex> = assimp_mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let normal = assimp_mesh
                .normals
                .get(i)
                .map(|n| [n.x, n.y, n.z])
                .unwrap_or([0.0, 0.0, 0.0]);

            let tex_coords = assimp_mesh
                .texture_coords
                .first()
                .and_then(|channel| channel.as_ref())
                .and_then(|coords| coords.get(i))
                .map(|tc| [tc.x, tc.y, 0.0])
                .unwrap_or([0.0, 0.0, 0.0]);

            Vertex {
                position: [v.x, v.y, v.z],
                tex_coords,
                normal,
            }
        })
        .collect();

    // Collect all face indices as u32
    let indices: Vec<u32> = assimp_mesh
        .faces
        .iter()
        .flat_map(|face| face.0.iter().copied())
        .collect();

    // Split if necessary for u16 index limit
    if vertices.len() <= MeshIndex::MAX as usize {
        // Simple case: fits in u16
        let u16_indices: Vec<MeshIndex> = indices.iter().map(|&i| i as MeshIndex).collect();
        let primitive = MeshPrimitive {
            primitive_type: PrimitiveType::TriangleList,
            indices: u16_indices,
        };
        let mesh = Mesh::from_raw(vertices, vec![primitive]);
        let mesh_id = scene.add_mesh(mesh);
        vec![(mesh_id, material_id)]
    } else {
        // Need to split into chunks
        let chunks = split_mesh(&vertices, &indices);
        chunks
            .into_iter()
            .map(|(chunk_verts, chunk_prim)| {
                let mesh = Mesh::from_raw(chunk_verts, vec![chunk_prim]);
                let mesh_id = scene.add_mesh(mesh);
                (mesh_id, material_id)
            })
            .collect()
    }
}

/// Split a mesh with >65535 vertices into multiple chunks that each fit in u16 indices.
fn split_mesh(vertices: &[Vertex], indices: &[u32]) -> Vec<(Vec<Vertex>, MeshPrimitive)> {
    let max_verts = MeshIndex::MAX as usize;
    let mut chunks: Vec<(Vec<Vertex>, MeshPrimitive)> = Vec::new();

    let mut chunk_verts: Vec<Vertex> = Vec::new();
    let mut chunk_indices: Vec<MeshIndex> = Vec::new();
    let mut remap: HashMap<u32, MeshIndex> = HashMap::new();

    for triangle in indices.chunks(3) {
        if triangle.len() < 3 {
            break;
        }

        // Check if adding this triangle would overflow the chunk
        let new_verts_needed = triangle
            .iter()
            .filter(|&&idx| !remap.contains_key(&idx))
            .count();

        if chunk_verts.len() + new_verts_needed > max_verts {
            // Finalize current chunk
            if !chunk_indices.is_empty() {
                chunks.push((
                    std::mem::take(&mut chunk_verts),
                    MeshPrimitive {
                        primitive_type: PrimitiveType::TriangleList,
                        indices: std::mem::take(&mut chunk_indices),
                    },
                ));
            }
            remap.clear();
        }

        // Add triangle to current chunk
        for &orig_idx in triangle {
            let local_idx = *remap.entry(orig_idx).or_insert_with(|| {
                let idx = chunk_verts.len() as MeshIndex;
                chunk_verts.push(vertices[orig_idx as usize]);
                idx
            });
            chunk_indices.push(local_idx);
        }
    }

    // Finalize last chunk
    if !chunk_indices.is_empty() {
        chunks.push((
            chunk_verts,
            MeshPrimitive {
                primitive_type: PrimitiveType::TriangleList,
                indices: chunk_indices,
            },
        ));
    }

    chunks
}

// ============================================================================
// Node Tree
// ============================================================================

/// Recursively build the scene node tree from the assimp node hierarchy.
fn build_node_tree(
    node: &Rc<RNode>,
    scene: &mut Scene,
    mesh_map: &[MeshEntry],
    material_map: &[MaterialId],
    parent: Option<NodeId>,
) -> Result<()> {
    // Decompose the assimp 4x4 transform matrix
    let t = &node.transformation;
    #[rustfmt::skip]
    let matrix = Matrix4::new(
        t.a1, t.b1, t.c1, t.d1,
        t.a2, t.b2, t.c2, t.d2,
        t.a3, t.b3, t.c3, t.d3,
        t.a4, t.b4, t.c4, t.d4,
    );

    let (position, rotation, scale) = decompose_matrix(&matrix);

    let name = if node.name.is_empty() {
        None
    } else {
        Some(node.name.clone())
    };

    if node.meshes.is_empty() {
        // Pure transform node (no geometry)
        let node_id = scene.add_node(parent, name, position, rotation, scale)?;

        // Recurse into children
        for child in node.children.borrow().iter() {
            build_node_tree(child, scene, mesh_map, material_map, Some(node_id))?;
        }
    } else {
        // Node references one or more meshes
        if node.meshes.len() == 1 {
            // Single mesh: create one instance node
            let mesh_idx = node.meshes[0] as usize;
            if let Some(entries) = mesh_map.get(mesh_idx) {
                if entries.len() == 1 {
                    let (mesh_id, mat_id) = entries[0];
                    let node_id = scene.add_instance_node(
                        parent, mesh_id, mat_id, name, position, rotation, scale,
                    )?;

                    for child in node.children.borrow().iter() {
                        build_node_tree(child, scene, mesh_map, material_map, Some(node_id))?;
                    }
                } else {
                    // Split mesh: create a parent transform node, then instance children
                    let group_id = scene.add_node(parent, name, position, rotation, scale)?;
                    for (i, &(mesh_id, mat_id)) in entries.iter().enumerate() {
                        let chunk_name = Some(format!("chunk_{}", i));
                        scene.add_instance_node(
                            Some(group_id),
                            mesh_id,
                            mat_id,
                            chunk_name,
                            Point3::new(0.0, 0.0, 0.0),
                            Quaternion::new(1.0, 0.0, 0.0, 0.0),
                            Vector3::new(1.0, 1.0, 1.0),
                        )?;
                    }

                    for child in node.children.borrow().iter() {
                        build_node_tree(child, scene, mesh_map, material_map, Some(group_id))?;
                    }
                }
            }
        } else {
            // Multiple meshes on one node: create parent + instance children
            let group_id = scene.add_node(parent, name, position, rotation, scale)?;

            for &mesh_idx in &node.meshes {
                if let Some(entries) = mesh_map.get(mesh_idx as usize) {
                    for &(mesh_id, mat_id) in entries {
                        scene.add_instance_node(
                            Some(group_id),
                            mesh_id,
                            mat_id,
                            None,
                            Point3::new(0.0, 0.0, 0.0),
                            Quaternion::new(1.0, 0.0, 0.0, 0.0),
                            Vector3::new(1.0, 1.0, 1.0),
                        )?;
                    }
                }
            }

            for child in node.children.borrow().iter() {
                build_node_tree(child, scene, mesh_map, material_map, Some(group_id))?;
            }
        }
    }

    Ok(())
}

// ============================================================================
// Lights
// ============================================================================

/// Load lights from assimp scene into our scene.
fn load_lights(assimp_lights: &[russimp::light::Light], scene: &mut Scene) {
    use russimp::light::LightSourceType;

    for light in assimp_lights.iter().take(crate::light::MAX_LIGHTS) {
        let color = RgbaColor {
            r: light.color_diffuse.r,
            g: light.color_diffuse.g,
            b: light.color_diffuse.b,
            a: 1.0,
        };

        // Estimate intensity from color magnitude (assimp doesn't have a separate intensity field)
        let intensity = 1.0;

        let engine_light = match light.light_source_type {
            LightSourceType::Point => Light::point(
                Vector3::new(light.pos.x, light.pos.y, light.pos.z),
                color,
                intensity,
            ),
            LightSourceType::Directional => Light::directional(
                Vector3::new(light.direction.x, light.direction.y, light.direction.z),
                color,
                intensity,
            ),
            LightSourceType::Spot => Light::spot(
                Vector3::new(light.pos.x, light.pos.y, light.pos.z),
                Vector3::new(light.direction.x, light.direction.y, light.direction.z),
                color,
                intensity,
                light.angle_inner_cone,
                light.angle_outer_cone,
            ),
            _ => continue, // Skip Ambient, Area, Undefined
        };

        scene.lights.push(engine_light);
    }
}

// ============================================================================
// Camera
// ============================================================================

/// Extract the first camera from the assimp scene, if any.
fn extract_camera(cameras: &[russimp::camera::Camera]) -> Option<Camera> {
    let cam = cameras.first()?;

    let eye = Point3::new(cam.position.x, cam.position.y, cam.position.z);
    let look_at = Vector3::new(cam.look_at.x, cam.look_at.y, cam.look_at.z);
    let target = Point3::new(eye.x + look_at.x, eye.y + look_at.y, eye.z + look_at.z);
    let up = Vector3::new(cam.up.x, cam.up.y, cam.up.z);

    // Assimp provides horizontal FOV in radians; convert to vertical FOV in degrees
    let aspect = if cam.aspect > 0.0 {
        cam.aspect
    } else {
        16.0 / 9.0
    };
    let fovy_rad = 2.0 * ((cam.horizontal_fov / 2.0).tan() / aspect).atan();
    let fovy = fovy_rad.to_degrees();

    let znear = if cam.clip_plane_near > 0.0 {
        cam.clip_plane_near
    } else {
        0.1
    };
    let zfar = if cam.clip_plane_far > 0.0 {
        cam.clip_plane_far
    } else {
        1000.0
    };

    Some(Camera {
        eye,
        target,
        up,
        aspect,
        fovy,
        znear,
        zfar,
        ortho: false,
    })
}

// ============================================================================
// Format Support
// ============================================================================

/// File extensions that assimp can handle (beyond what we already support natively).
///
/// Note: glTF (.glb/.gltf) is intentionally excluded — our native loader is preferred.
pub const ASSIMP_EXTENSIONS: &[&str] = &[
    "obj", "fbx", "dae", "blend", "3ds", "ase", "ifc", "xgl", "zgl", "ply", "dxf", "lwo", "lws",
    "lxo", "stl", "x", "ac", "ms3d", "cob", "scn", "bvh", "csm", "irrmesh", "irr", "mdl",
    "md2", "md3", "pk3", "mdc", "md5mesh", "md5anim", "md5camera", "smd", "vta", "ogex", "3d",
    "b3d", "q3d", "q3s", "nff", "off", "raw", "ter", "hmp", "ndo", "stp", "step",
];

/// Check if a file extension is one that assimp handles.
pub fn is_assimp_extension(ext: &str) -> bool {
    ASSIMP_EXTENSIONS.contains(&ext.to_lowercase().as_str())
}
