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

/// Tries to resolve an external URI image to a file path.
fn resolve_image_path(
    image_index: usize,
    document: &gltf::Document,
    base_path: Option<&Path>,
) -> Option<std::path::PathBuf> {
    let base = base_path?;
    let img = document.images().nth(image_index)?;
    let gltf::image::Source::Uri { uri, .. } = img.source() else { return None };

    // Skip data: URIs - those are embedded
    if uri.starts_with("data:") {
        return None;
    }

    let path = base.join(uri);
    path.exists().then_some(path)
}

/// Extracts original compressed bytes from a glTF buffer view.
fn extract_embedded_image_bytes(
    image_index: usize,
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Option<(Vec<u8>, crate::scene::format::TextureFormat)> {
    use crate::scene::format::TextureFormat;

    let img = document.images().nth(image_index)?;
    let gltf::image::Source::View { view, mime_type } = img.source() else { return None };

    let format = match mime_type {
        "image/png" => TextureFormat::Png,
        "image/jpeg" => TextureFormat::Jpeg,
        _ => return None,
    };

    let buffer_data = buffers.get(view.buffer().index())?;
    let start = view.offset();
    let end = start + view.length();
    let bytes = buffer_data.get(start..end)?.to_vec();

    Some((bytes, format))
}

/// Decodes glTF image pixel data into a DynamicImage.
fn decode_gltf_image(gltf_image: &gltf::image::Data) -> anyhow::Result<image::DynamicImage> {
    match gltf_image.format {
        gltf::image::Format::R8G8B8A8 => {
            let img = image::RgbaImage::from_raw(
                gltf_image.width,
                gltf_image.height,
                gltf_image.pixels.clone(),
            )
            .ok_or_else(|| anyhow::anyhow!("Failed to create RGBA image from glTF data"))?;
            Ok(image::DynamicImage::ImageRgba8(img))
        }
        gltf::image::Format::R8G8B8 => {
            let img = image::RgbImage::from_raw(
                gltf_image.width,
                gltf_image.height,
                gltf_image.pixels.clone(),
            )
            .ok_or_else(|| anyhow::anyhow!("Failed to create RGB image from glTF data"))?;
            Ok(image::DynamicImage::ImageRgb8(img))
        }
        _ => Err(anyhow::anyhow!("Unsupported glTF image format: {:?}", gltf_image.format)),
    }
}

/// Loads an image from glTF image data and adds it as a texture to the scene.
///
/// Preserves original compressed bytes when loading from embedded buffer views,
/// or uses path-based textures for external URI references.
fn load_gltf_texture(
    gltf_image: &gltf::image::Data,
    image_index: usize,
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    base_path: Option<&Path>,
    scene: &mut Scene,
) -> anyhow::Result<crate::scene::TextureId> {
    // For external files, use path-based texture (lazy loading, preserves original format)
    if let Some(path) = resolve_image_path(image_index, document, base_path) {
        let texture = Texture::from_path(path);
        return Ok(scene.add_texture(texture));
    }

    // Decode the image from glTF pixel data
    let dynamic_image = decode_gltf_image(gltf_image)?;

    // For embedded images, try to preserve original compressed bytes
    let texture = match extract_embedded_image_bytes(image_index, document, buffers) {
        Some((bytes, format)) => Texture::from_image_with_original_bytes(dynamic_image, bytes, format),
        None => Texture::from_image(dynamic_image),
    };

    Ok(scene.add_texture(texture))
}

/// Loads a material from a glTF material definition.
///
/// Extracts full PBR data including base color, metallic-roughness, and normal maps.
/// Returns the MaterialId of the created material.
fn load_material(
    gltf_material: &gltf::Material,
    images: &[gltf::image::Data],
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    base_path: Option<&Path>,
    scene: &mut Scene,
) -> anyhow::Result<MaterialId> {
    let pbr = gltf_material.pbr_metallic_roughness();

    // Start with a new material
    let mut material = Material::new();

    // Base color factor (always present, defaults to white in glTF)
    let base_color = pbr.base_color_factor();
    material = material.with_base_color_factor(crate::common::RgbaColor {
        r: base_color[0],
        g: base_color[1],
        b: base_color[2],
        a: base_color[3],
    });

    // Base color texture (optional)
    if let Some(tex_info) = pbr.base_color_texture() {
        let image_index = tex_info.texture().source().index();
        let texture_id = load_gltf_texture(&images[image_index], image_index, document, buffers, base_path, scene)?;
        material = material.with_base_color_texture(texture_id);
    }

    // Metallic and roughness factors
    material = material
        .with_metallic_factor(pbr.metallic_factor())
        .with_roughness_factor(pbr.roughness_factor());

    // Metallic-roughness texture (optional)
    // glTF packs roughness in G channel and metallic in B channel
    if let Some(tex_info) = pbr.metallic_roughness_texture() {
        let image_index = tex_info.texture().source().index();
        let texture_id = load_gltf_texture(&images[image_index], image_index, document, buffers, base_path, scene)?;
        material = material.with_metallic_roughness_texture(texture_id);
    }

    // Normal map (from material, not from PBR extension)
    if let Some(normal_tex) = gltf_material.normal_texture() {
        let image_index = normal_tex.texture().source().index();
        let texture_id = load_gltf_texture(&images[image_index], image_index, document, buffers, base_path, scene)?;
        material = material
            .with_normal_texture(texture_id)
            .with_normal_scale(normal_tex.scale());
    }

    let mat_id = scene.add_material(material);
    Ok(mat_id)
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
                ortho: false,
            })
        }
        gltf::camera::Projection::Orthographic(ortho_cam) => {
            // glTF orthographic uses xmag/ymag as half-extents
            // We derive an equivalent fovy from the distance and ymag
            let ymag = ortho_cam.ymag();
            let znear = ortho_cam.znear();
            let zfar = ortho_cam.zfar();

            // Calculate an equivalent fovy that would produce the same view height
            // at the current distance: ymag = distance * tan(fovy/2)
            // So: fovy = 2 * atan(ymag / distance)
            let distance = (eye - target).magnitude();
            let fovy = if distance > 0.0 {
                2.0 * (ymag / distance).atan().to_degrees()
            } else {
                45.0 // fallback
            };

            Some(Camera {
                eye,
                target,
                up,
                aspect,
                fovy,
                znear,
                zfar,
                ortho: true,
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
/// # Arguments
/// * `path` - Path to the glTF file
/// * `aspect` - Aspect ratio to use for the camera (if found)
pub fn load_gltf_scene_from_path<P: AsRef<Path>>(
    path: P,
    aspect: f32,
) -> anyhow::Result<GltfLoadResult> {
    let path = path.as_ref();
    let base_path = path.parent().map(|p| p.to_path_buf());
    let (document, buffers, images) = gltf::import(path)?;
    load_gltf_from_data(document, buffers, images, base_path.as_deref(), aspect)
}

/// Loads a glTF scene from a byte slice.
///
/// This is useful for loading glTF data that has been read from memory,
/// such as when loading files in a web browser via JavaScript.
/// Returns a `GltfLoadResult` containing the scene and an optional camera
/// if one was defined in the glTF file.
///
/// # Arguments
/// * `data` - The glTF file contents as a byte slice
/// * `aspect` - Aspect ratio to use for the camera (if found)
pub fn load_gltf_scene_from_slice(
    data: &[u8],
    aspect: f32,
) -> anyhow::Result<GltfLoadResult> {
    let (document, buffers, images) = gltf::import_slice(data)?;
    load_gltf_from_data(document, buffers, images, None, aspect)
}

/// Internal helper to load a glTF scene from parsed data.
fn load_gltf_from_data(
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    images: Vec<gltf::image::Data>,
    base_path: Option<&Path>,
    aspect: f32,
) -> anyhow::Result<GltfLoadResult> {
    use std::collections::HashMap;
    use cgmath::SquareMatrix;

    let mut scene = Scene::new();

    // Load all materials first (they'll be added to scene.materials)
    let mut material_map: Vec<MaterialId> = Vec::new();
    for material in document.materials() {
        let mat_id = load_material(&material, &images, &document, &buffers, base_path, &mut scene)?;
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
            load_node_recursive(&gltf_node, None, &mut scene, &mesh_map, &mut node_map)?;

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

    // Add default lights if the glTF scene doesn't define any
    scene.set_default_lights();

    Ok(GltfLoadResult { scene, camera })
}

/// Recursively loads a glTF node and its children.
fn load_node_recursive(
    gltf_node: &gltf::Node,
    parent: Option<crate::scene::NodeId>,
    scene: &mut crate::scene::Scene,
    mesh_map: &[Vec<LoadedPrimitive>],
    node_map: &mut std::collections::HashMap<usize, crate::scene::NodeId>,
) -> anyhow::Result<()> {

    // Decompose transform
    let (position, rotation, scale) = decompose_transform(&gltf_node.transform());

    // Extract node name from glTF
    let name = gltf_node.name().map(|s| s.to_string());

    // Create node(s) and track the parent node ID for child recursion
    let node_id = if let Some(mesh) = gltf_node.mesh() {
        let primitives = &mesh_map[mesh.index()];

        if primitives.is_empty() {
            // Mesh has no triangle primitives (only lines/points), treat as transform node
            scene.add_node(parent, name, position, rotation, scale)?
        } else if primitives.len() > 1 {
            // Multi-primitive mesh: create parent node and child nodes for each primitive
            let parent_node_id = scene.add_node(parent, name, position, rotation, scale)?;

            // Create child nodes for each primitive with identity transform
            for (mesh_id, material_id) in primitives {
                scene.add_instance_node(
                    Some(parent_node_id),
                    *mesh_id,
                    *material_id,
                    None,
                    cgmath::Point3::new(0.0, 0.0, 0.0),
                    cgmath::Quaternion::new(1.0, 0.0, 0.0, 0.0),
                    cgmath::Vector3::new(1.0, 1.0, 1.0),
                )?;
            }

            parent_node_id
        } else {
            // Single primitive: create a single instance node
            let (mesh_id, material_id) = primitives[0];
            scene.add_instance_node(parent, mesh_id, material_id, name, position, rotation, scale)?
        }
    } else {
        // No mesh, just a transform node
        scene.add_node(parent, name, position, rotation, scale)?
    };

    node_map.insert(gltf_node.index(), node_id);

    // Recurse for all children
    for child in gltf_node.children() {
        load_node_recursive(&child, Some(node_id), scene, mesh_map, node_map)?;
    }

    Ok(())
}