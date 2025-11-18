use std::path::Path;
use crate::{material::DefaultMaterial, scene::{MeshPrimitive, PrimitiveType, Vertex}};

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
    material_map: &[crate::material::MaterialId],
) -> crate::material::MaterialId {
    match primitive_type {
        PrimitiveType::TriangleList => {
            // For triangles, use the glTF material if available
            primitive
                .material()
                .index()
                .and_then(|idx| material_map.get(idx).copied())
                .unwrap_or(DefaultMaterial::Face.into())
        }
        PrimitiveType::LineList => DefaultMaterial::Line.into(),
        PrimitiveType::PointList => DefaultMaterial::Point.into(),
    }
}

/// Loads a material from a glTF material definition.
///
/// Converts glTF PBR materials to either ColorMaterial or TextureMaterial.
fn load_material(
    gltf_material: &gltf::Material,
    images: &[gltf::image::Data],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    material_manager: &mut crate::material::MaterialManager,
) -> anyhow::Result<crate::material::MaterialId> {
    use crate::texture::Texture;

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

        // Create texture from image data
        let texture = Texture::from_image(device, queue, &dynamic_image, Some("gltf_texture"))?;

        // Create TextureMaterial
        let mat_id = material_manager.create_face_texture_material(device, texture)?;
        Ok(mat_id)
    } else {
        // No texture, use base color factor as ColorMaterial
        let base_color = pbr.base_color_factor();
        let color = crate::common::RgbaColor {
            r: base_color[0],
            g: base_color[1],
            b: base_color[2],
            a: base_color[3],
        };

        Ok(material_manager.create_face_color_material(device, color))
    }
}

/// Decomposes a glTF transform into position, rotation, and scale.
fn decompose_transform(transform: &gltf::scene::Transform) -> (cgmath::Point3<f32>, cgmath::Quaternion<f32>, cgmath::Vector3<f32>) {
    use cgmath::{Matrix4, Point3, Quaternion, Vector3};

    match transform {
        gltf::scene::Transform::Matrix { matrix } => {
            // glTF uses column-major order, same as cgmath
            let cgmath_matrix = Matrix4::from(*matrix);
            // Use the decompose_matrix function from common.rs
            crate::common::decompose_matrix(&cgmath_matrix)
        }
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => {
            let pos = Point3::from(*translation);
            let rot = Quaternion::new(rotation[3], rotation[0], rotation[1], rotation[2]); // glTF: [x,y,z,w], cgmath: w,x,y,z
            let scl = Vector3::from(*scale);
            (pos, rot, scl)
        }
    }
}

/// Loads a glTF scene from a file.
///
/// This is the main entry point for loading glTF files into the engine.
pub fn load_gltf_scene<P: AsRef<Path>>(
    path: P,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    material_manager: &mut crate::material::MaterialManager,
) -> anyhow::Result<crate::scene::Scene> {
    use crate::scene::{MeshDescriptor, Scene};
    use std::collections::HashMap;

    let (document, buffers, images) = gltf::import(path)?;

    let mut scene = Scene::new();

    // Load all materials first
    let mut material_map: Vec<crate::material::MaterialId> = Vec::new();
    for material in document.materials() {
        let mat_id = load_material(&material, &images, device, queue, material_manager)?;
        material_map.push(mat_id);
    }

    // Maps glTF mesh index -> list of (scene mesh, material) pairs
    // We need this structure because glTF nodes reference mesh indices, and each glTF mesh
    // can contain multiple primitives. We load each primitive as a separate scene mesh.
    let mut mesh_map: Vec<Vec<(crate::scene::MeshId, crate::material::MaterialId)>> = Vec::new();

    for mesh in document.meshes() {
        let mut primitives_data = Vec::new();

        for (prim_idx, primitive) in mesh.primitives().enumerate() {
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

            // Create mesh from raw data
            let mesh_label = format!(
                "{}_{}_prim_{}",
                mesh.name().unwrap_or("mesh"),
                mesh.index(),
                prim_idx
            );

            let primitives = vec![MeshPrimitive {
                primitive_type,
                indices,
            }];

            let mesh_descriptor = MeshDescriptor::Raw { vertices, primitives };
            let mesh_id = scene.add_mesh(device, mesh_descriptor, Some(&mesh_label))?;

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

    // Load the scene hierarchy
    if let Some(gltf_scene) = document.default_scene().or_else(|| document.scenes().next()) {
        // Map from glTF node index to our NodeId
        let mut node_map: HashMap<usize, crate::scene::NodeId> = HashMap::new();

        // Process all root nodes
        for gltf_node in gltf_scene.nodes() {
            load_node_recursive(&gltf_node, None, &mut scene, &mesh_map, &mut node_map);
        }
    }

    Ok(scene)
}

/// Recursively loads a glTF node and its children.
fn load_node_recursive(
    gltf_node: &gltf::Node,
    parent: Option<crate::scene::NodeId>,
    scene: &mut crate::scene::Scene,
    mesh_map: &[Vec<(crate::scene::MeshId, crate::material::MaterialId)>],
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