use std::path::Path;
use crate::{
    camera::Camera,
    scene::{
        Material, Texture, Mesh, MeshId, MeshPrimitive, PrimitiveType, Scene, Vertex,
        MaterialId, DEFAULT_MATERIAL_ID,
    },
};

/// A loaded primitive from a glTF mesh, containing the scene mesh ID and its material ID.
type LoadedPrimitive = (MeshId, MaterialId);

/// Maps glTF mesh indices to their loaded primitives.
///
/// Each glTF mesh can contain multiple primitives (e.g. different parts with different materials).
/// Each primitive is loaded as a separate scene mesh, so a single glTF mesh index maps to
/// potentially multiple (MeshId, MaterialId) pairs.
type GltfMeshMap = Vec<Vec<LoadedPrimitive>>;

/// Result of loading a glTF scene, containing the scene and optional camera.
pub struct GltfLoadResult {
    /// The loaded scene with meshes, materials, and scene tree
    pub scene: Scene,
    /// Camera from the glTF file, if one was defined
    pub camera: Option<Camera>,
}

/// Loads vertex data from a glTF primitive.
///
/// Reads positions, normals, and texture coordinates from the primitive's attributes.
fn load_vertices(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
) -> anyhow::Result<Vec<Vertex>> {
    // Create a reader for this primitive
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    // Read positions (required)
    let positions = reader
        .read_positions()
        .ok_or_else(|| anyhow::anyhow!("Primitive missing positions"))?;

    // Read normals (optional - lines and points don't need them)
    let normals = reader.read_normals();

    // Read texture coordinates (optional, default to [0, 0] if missing)
    let tex_coords = reader.read_tex_coords(0);

    // Build vertex array by zipping iterators
    let vertices = match (normals, tex_coords) {
        (Some(normals), Some(tex_coords)) => {
            // Has normals and texture coordinates
            let tex_coords_f32 = tex_coords.into_f32();
            positions
                .zip(normals)
                .zip(tex_coords_f32)
                .map(|((position, normal), tex_coords)| Vertex {
                    position,
                    normal,
                    tex_coords: [tex_coords[0], tex_coords[1], 0.0],
                })
                .collect()
        }
        (Some(normals), None) => {
            // Has normals but no texture coordinates
            positions
                .zip(normals)
                .map(|(position, normal)| Vertex {
                    position,
                    normal,
                    tex_coords: [0.0, 0.0, 0.0],
                })
                .collect()
        }
        (None, Some(tex_coords)) => {
            // Has texture coordinates but no normals
            let tex_coords_f32 = tex_coords.into_f32();
            positions
                .zip(tex_coords_f32)
                .map(|(position, tex_coords)| Vertex {
                    position,
                    normal: [0.0, 0.0, 0.0],
                    tex_coords: [tex_coords[0], tex_coords[1], 0.0],
                })
                .collect()
        }
        (None, None) => {
            // No normals or texture coordinates (lines/points)
            positions
                .map(|position| Vertex {
                    position,
                    normal: [0.0, 0.0, 0.0],
                    tex_coords: [0.0, 0.0, 0.0],
                })
                .collect()
        }
    };

    Ok(vertices)
}

/// Loads index data from a glTF primitive.
fn load_indices(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
) -> anyhow::Result<Vec<u16>> {
    // Create a reader for this primitive
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    // Read indices and convert to u16
    let indices = reader
        .read_indices()
        .ok_or_else(|| anyhow::anyhow!("Primitive missing indices"))?
        .into_u32()
        .map(|i| i as u16) // Convert u32 to u16 (assumes small meshes)
        .collect();

    Ok(indices)
}

/// Maps a glTF primitive mode to our PrimitiveType.
/// Returns None for unsupported modes.
fn map_primitive_mode(mode: gltf::mesh::Mode) -> Option<PrimitiveType> {
    match mode {
        gltf::mesh::Mode::Triangles => Some(PrimitiveType::TriangleList),
        gltf::mesh::Mode::Lines => Some(PrimitiveType::LineList),
        gltf::mesh::Mode::Points => Some(PrimitiveType::PointList),
        _ => None, // TriangleFan, TriangleStrip not supported
    }
}

/// Gets an appropriate material for a glTF primitive based on its type.
fn get_material_for_primitive(
    primitive: &gltf::Primitive,
    primitive_type: PrimitiveType,
    material_map: &[MaterialId],
) -> MaterialId {
    match primitive_type {
        PrimitiveType::TriangleList => {
            // For triangles, use the glTF material if available
            primitive
                .material()
                .index()
                .and_then(|idx| material_map.get(idx).copied())
                .unwrap_or(DEFAULT_MATERIAL_ID)
        }
        PrimitiveType::LineList | PrimitiveType::PointList => DEFAULT_MATERIAL_ID,
    }
}

/// Loads a material from a glTF material definition.
///
/// Converts glTF PBR materials to either color or textured materials.
/// Returns the MaterialId of the created material.
fn load_material(
    gltf_material: &gltf::Material,
    images: &[gltf::image::Data],
    scene: &mut Scene,
) -> anyhow::Result<MaterialId> {
    let pbr = gltf_material.pbr_metallic_roughness();

    // Check if material has a base color texture
    if let Some(gltf_texture_info) = pbr.base_color_texture() {
        let gltf_texture = gltf_texture_info.texture();
        let image_index = gltf_texture.source().index();
        let gltf_image = &images[image_index];

        let dynamic_image = match gltf_image.format {
            gltf::image::Format::R8G8B8A8 => {
                image::DynamicImage::ImageRgba8(
                    image::RgbaImage::from_raw(
                        gltf_image.width,
                        gltf_image.height,
                        gltf_image.pixels.clone()
                    ).ok_or_else(|| anyhow::anyhow!("Failed to create image from glTF data"))?
                )
            }
            gltf::image::Format::R8G8B8 => {
                image::DynamicImage::ImageRgb8(
                    image::RgbImage::from_raw(
                        gltf_image.width,
                        gltf_image.height,
                        gltf_image.pixels.clone()
                    ).ok_or_else(|| anyhow::anyhow!("Failed to create image from glTF data"))?
                )
            }
            _ => {
                return Err(anyhow::anyhow!("Unsupported glTF image format"));
            }
        };

        // Create texture from image data (no GPU resources yet - lazy initialization)
        let texture = Texture::from_image(dynamic_image);
        let texture_id = scene.add_texture(texture);

        // Create material with face texture reference
        let material = Material::new().with_face_texture(texture_id);
        let mat_id = scene.add_material(material);
        Ok(mat_id)
    } else {
        // No texture, use base color factor
        let base_color = pbr.base_color_factor();
        let color = crate::common::RgbaColor {
            r: base_color[0],
            g: base_color[1],
            b: base_color[2],
            a: base_color[3],
        };

        // Create material with face color
        let material = Material::new().with_face_color(color);
        let mat_id = scene.add_material(material);
        Ok(mat_id)
    }
}

/// Converts a glTF transform to a 4x4 matrix.
fn transform_to_matrix(transform: &gltf::scene::Transform) -> cgmath::Matrix4<f32> {
    cgmath::Matrix4::from(transform.clone().matrix())
}

/// Decomposes a glTF transform into position, rotation, and scale.
fn decompose_transform(transform: &gltf::scene::Transform) -> (cgmath::Point3<f32>, cgmath::Quaternion<f32>, cgmath::Vector3<f32>) {
    crate::common::decompose_matrix(&transform_to_matrix(transform))
}

/// Extracts camera data from a glTF camera node.
///
/// Returns a Camera with the appropriate parameters if the node contains a camera.
fn extract_camera_from_node(
    gltf_node: &gltf::Node,
    world_transform: cgmath::Matrix4<f32>,
    aspect: f32,
) -> Option<Camera> {
    use cgmath::{InnerSpace, Point3, Vector3};

    let gltf_camera = gltf_node.camera()?;

    // Extract camera position from world transform (translation column)
    let eye = Point3::new(
        world_transform[3][0],
        world_transform[3][1],
        world_transform[3][2],
    );

    // Extract forward direction from world transform
    // In glTF, cameras look down -Z in their local space
    // The third column of the rotation matrix gives us the local Z axis
    let local_z = Vector3::new(
        world_transform[2][0],
        world_transform[2][1],
        world_transform[2][2],
    ).normalize();

    // Camera looks down -Z, so forward is -local_z
    let forward = -local_z;

    // Extract up direction from world transform (Y axis)
    let up = Vector3::new(
        world_transform[1][0],
        world_transform[1][1],
        world_transform[1][2],
    ).normalize();

    // Target is some distance in front of the camera
    let target = eye + forward;

    match gltf_camera.projection() {
        gltf::camera::Projection::Perspective(persp) => {
            // glTF perspective uses yfov in radians
            let fovy = persp.yfov().to_degrees();
            let znear = persp.znear();
            // glTF may not specify zfar (infinite projection)
            let zfar = persp.zfar().unwrap_or(1000.0);

            Some(Camera {
                eye,
                target,
                up,
                aspect,
                fovy,
                znear,
                zfar,
            })
        }
        gltf::camera::Projection::Orthographic(_ortho) => {
            // We don't support orthographic cameras yet, but we can still create
            // a perspective camera at the same position as a fallback
            log::warn!("Orthographic camera found but not supported, using perspective fallback");
            Some(Camera {
                eye,
                target,
                up,
                aspect,
                fovy: 45.0,
                znear: 0.1,
                zfar: 1000.0,
            })
        }
    }
}

/// Recursively searches for a camera in the glTF scene tree.
///
/// Returns the first camera found along with its world transform.
fn find_camera_recursive(
    gltf_node: &gltf::Node,
    parent_transform: cgmath::Matrix4<f32>,
    aspect: f32,
) -> Option<Camera> {
    let local_transform = transform_to_matrix(&gltf_node.transform());
    let world_transform = parent_transform * local_transform;

    // Check if this node has a camera
    if let Some(camera) = extract_camera_from_node(gltf_node, world_transform, aspect) {
        return Some(camera);
    }

    // Recurse into children
    for child in gltf_node.children() {
        if let Some(camera) = find_camera_recursive(&child, world_transform, aspect) {
            return Some(camera);
        }
    }

    None
}

/// Loads a glTF scene from a file.
///
/// This is the main entry point for loading glTF files into the engine.
/// Returns a `GltfLoadResult` containing the scene and an optional camera
/// if one was defined in the glTF file.
///
/// GPU resources are not created during loading - they are lazily initialized
/// when the scene is first rendered via `DrawState::prepare_scene()`.
///
/// # Arguments
/// * `path` - Path to the glTF file
/// * `aspect` - Aspect ratio to use for the camera (if found)
pub fn load_gltf_scene<P: AsRef<Path>>(
    path: P,
    aspect: f32,
) -> anyhow::Result<GltfLoadResult> {
    use std::collections::HashMap;
    use cgmath::SquareMatrix;

    let (document, buffers, images) = gltf::import(path)?;

    let mut scene = Scene::new();

    // Load all materials first (they'll be added to scene.materials)
    let mut material_map: Vec<MaterialId> = Vec::new();
    for material in document.materials() {
        let mat_id = load_material(&material, &images, &mut scene)?;
        material_map.push(mat_id);
    }

    // Maps glTF mesh index -> list of (scene mesh, material) pairs
    // We need this structure because glTF nodes reference mesh indices, and each glTF mesh
    // can contain multiple primitives. We load each primitive as a separate scene mesh.
    let mut mesh_map: GltfMeshMap = Vec::new();

    for mesh in document.meshes() {
        let mut primitives_data = Vec::new();

        for (_prim_idx, primitive) in mesh.primitives().enumerate() {
            // Map glTF primitive mode to our PrimitiveType
            let Some(primitive_type) = map_primitive_mode(primitive.mode()) else {
                log::warn!(
                    "Skipping unsupported primitive mode {:?} in mesh {}",
                    primitive.mode(),
                    mesh.name().unwrap_or("unnamed")
                );
                continue;
            };

            // Load vertex and index data
            let vertices = load_vertices(&primitive, &buffers)?;
            let indices = load_indices(&primitive, &buffers)?;

            let primitives = vec![MeshPrimitive {
                primitive_type,
                indices,
            }];

            let mesh_obj = Mesh::from_raw(vertices, primitives);
            let mesh_id = scene.add_mesh(mesh_obj);

            // Create appropriate material for this primitive type
            let material_id = get_material_for_primitive(
                &primitive,
                primitive_type,
                &material_map,
            );

            primitives_data.push((mesh_id, material_id));
        }

        mesh_map.push(primitives_data);
    }

    // Search for a camera in the scene
    let mut camera: Option<Camera> = None;

    // Load the scene hierarchy
    if let Some(gltf_scene) = document.default_scene().or_else(|| document.scenes().next()) {
        // Map from glTF node index to our NodeId
        let mut node_map: HashMap<usize, crate::scene::NodeId> = HashMap::new();

        // Process all root nodes
        for gltf_node in gltf_scene.nodes() {
            load_node_recursive(&gltf_node, None, &mut scene, &mesh_map, &mut node_map);

            // Search for camera if we haven't found one yet
            if camera.is_none() {
                camera = find_camera_recursive(
                    &gltf_node,
                    cgmath::Matrix4::identity(),
                    aspect,
                );
            }
        }
    }

    Ok(GltfLoadResult { scene, camera })
}

/// Recursively loads a glTF node and its children.
fn load_node_recursive(
    gltf_node: &gltf::Node,
    parent: Option<crate::scene::NodeId>,
    scene: &mut crate::scene::Scene,
    mesh_map: &[Vec<LoadedPrimitive>],
    node_map: &mut std::collections::HashMap<usize, crate::scene::NodeId>,
) {

    // Decompose transform
    let (position, rotation, scale) = decompose_transform(&gltf_node.transform());

    // Create node(s) and track the parent node ID for child recursion
    let node_id = if let Some(mesh) = gltf_node.mesh() {
        let primitives = &mesh_map[mesh.index()];

        if primitives.is_empty() {
            // Mesh has no triangle primitives (only lines/points), treat as transform node
            scene.add_node(parent, position, rotation, scale)
        } else if primitives.len() > 1 {
            // Multi-primitive mesh: create parent node and child nodes for each primitive
            let parent_node_id = scene.add_node(parent, position, rotation, scale);

            // Create child nodes for each primitive with identity transform
            for (mesh_id, material_id) in primitives {
                scene.add_instance_node(
                    Some(parent_node_id),
                    *mesh_id,
                    *material_id,
                    cgmath::Point3::new(0.0, 0.0, 0.0),
                    cgmath::Quaternion::new(1.0, 0.0, 0.0, 0.0),
                    cgmath::Vector3::new(1.0, 1.0, 1.0),
                );
            }

            parent_node_id
        } else {
            // Single primitive: create a single instance node
            let (mesh_id, material_id) = primitives[0];
            scene.add_instance_node(parent, mesh_id, material_id, position, rotation, scale)
        }
    } else {
        // No mesh, just a transform node
        scene.add_node(parent, position, rotation, scale)
    };

    node_map.insert(gltf_node.index(), node_id);

    // Recurse for all children
    for child in gltf_node.children() {
        load_node_recursive(&child, Some(node_id), scene, mesh_map, node_map);
    }
}