use std::collections::HashMap;
use cgmath::{Point3, Vector3, Quaternion};
use crate::{
    common::RgbaColor,
    scene::{Material, Scene, NodeId, Mesh, MeshPrimitive, PrimitiveType, Vertex},
};

/// Unique identifier for annotations
pub type AnnotationId = u32;

/// Manages 3D annotations for debugging and visualization.
///
/// All annotations are stored under a single root node in the scene tree,
/// allowing for easy show/hide and bulk removal operations. The root node
/// is managed by the Scene and will be recreated if the scene is cleared.
///
/// After calling `Scene::clear()`, you should also call `AnnotationManager::reset()`
/// to clear stale annotation tracking state.
pub struct AnnotationManager {
    /// Maps annotation IDs to their scene node IDs
    annotation_nodes: HashMap<AnnotationId, NodeId>,

    /// Next annotation ID to assign
    next_annotation_id: AnnotationId,
}

impl AnnotationManager {
    /// Creates a new annotation manager.
    pub fn new() -> Self {
        Self {
            annotation_nodes: HashMap::new(),
            next_annotation_id: 0,
        }
    }

    /// Resets the annotation manager state.
    ///
    /// Call this after `Scene::clear()` to clear stale annotation tracking.
    pub fn reset(&mut self) {
        self.annotation_nodes.clear();
        self.next_annotation_id = 0;
    }

    /// Returns the number of active annotations.
    pub fn annotation_count(&self) -> usize {
        self.annotation_nodes.len()
    }

    /// Returns the root node ID for all annotations.
    ///
    /// This gets the annotation root from the scene, creating it if necessary.
    pub fn root_node(&self, scene: &mut Scene) -> NodeId {
        scene.annotation_root_node()
    }

    /// Shows or hides all annotations by scaling the root node.
    // TODO: Fix when proper visibility is implemented
    pub fn set_visible(&mut self, scene: &mut Scene, visible: bool) {
        let root_id = self.root_node(scene);
        if let Some(root) = scene.get_node_mut(root_id) {
            if visible {
                root.set_scale(Vector3::new(1.0, 1.0, 1.0));
            } else {
                root.set_scale(Vector3::new(0.0, 0.0, 0.0));
            }
        }
    }

    /// Removes a specific annotation by ID.
    pub fn remove_annotation(&mut self, scene: &mut Scene, id: AnnotationId) {
        if let Some(node_id) = self.annotation_nodes.remove(&id) {
            scene.remove_node(node_id);
        }
    }

    /// Clears all annotations.
    pub fn clear(&mut self, scene: &mut Scene) {
        for (_id, node_id) in self.annotation_nodes.drain() {
            scene.remove_node(node_id);
        }
    }

    /// Adds a line segment between two points.
    pub fn add_line(
        &mut self,
        scene: &mut Scene,
        start: Point3<f32>,
        end: Point3<f32>,
        color: RgbaColor,
    ) -> AnnotationId {
        self.add_polyline(scene, &[start, end], color, false)
    }

    /// Adds a polyline (connected line segments).
    pub fn add_polyline(
        &mut self,
        scene: &mut Scene,
        points: &[Point3<f32>],
        color: RgbaColor,
        closed: bool,
    ) -> AnnotationId {
        // Create vertices
        let vertices: Vec<Vertex> = points
            .iter()
            .map(|&p| Vertex {
                position: p.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
            })
            .collect();

        // Create line indices
        let mut indices = Vec::new();
        for i in 0..points.len().saturating_sub(1) {
            indices.push(i as u16);
            indices.push((i + 1) as u16);
        }

        // Close the loop if requested
        if closed && points.len() > 2 {
            indices.push((points.len() - 1) as u16);
            indices.push(0);
        }

        let primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices,
        }];

        // Get annotation root node first (before other borrows)
        let root_id = self.root_node(scene);

        // Create mesh
        let mesh = Mesh::from_raw(vertices, primitives);
        let mesh_id = scene.add_mesh(mesh);

        // Create material
        let material = Material::new().with_line_color(color);
        let material_id = scene.add_material(material);

        // Create node
        let node_id = scene.add_instance_node(
            Some(root_id),
            mesh_id,
            material_id,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        ).expect("Annotation root node must exist");

        // Track annotation
        let annotation_id = self.next_annotation_id;
        self.next_annotation_id += 1;
        self.annotation_nodes.insert(annotation_id, node_id);

        annotation_id
    }

    /// Adds coordinate axes at the specified origin.
    ///
    /// X axis is red, Y axis is green, Z axis is blue.
    pub fn add_axes(
        &mut self,
        scene: &mut Scene,
        origin: Point3<f32>,
        size: f32,
    ) -> AnnotationId {
        // Get annotation root node first (before other borrows)
        let root_id = self.root_node(scene);

        // Create parent node for axes group
        let parent_node_id = scene.add_default_node(Some(root_id))
            .expect("Annotation root node must exist");

        // Create X axis (red)
        let x_mat = scene.add_material(
            Material::new().with_line_color(RgbaColor { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }),
        );
        let x_start = origin;
        let x_end = origin + Vector3::new(size, 0.0, 0.0);
        let x_vertices = vec![
            Vertex {
                position: x_start.into(),
                tex_coords: [0.0; 3],
                normal: [1.0, 0.0, 0.0]
            },
            Vertex {
                position: x_end.into(),
                tex_coords: [0.0; 3],
                normal: [1.0, 0.0, 0.0]
            },
        ];
        let x_primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices: vec![0, 1]
        }];
        let x_mesh = scene.add_mesh(Mesh::from_raw(x_vertices, x_primitives));
        scene.add_instance_node(
            Some(parent_node_id),
            x_mesh,
            x_mat,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0)
        ).expect("Parent node must exist");

        // Create Y axis (green)
        let y_mat = scene.add_material(
            Material::new().with_line_color(RgbaColor { r: 0.0, g: 1.0, b: 0.0, a: 1.0 }),
        );
        let y_start = origin;
        let y_end = origin + Vector3::new(0.0, size, 0.0);
        let y_vertices = vec![
            Vertex {
                position: y_start.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0]
            },
            Vertex {
                position: y_end.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0]
            },
        ];
        let y_primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices: vec![0, 1]
        }];
        let y_mesh = scene.add_mesh(Mesh::from_raw(y_vertices, y_primitives));
        scene.add_instance_node(
            Some(parent_node_id),
            y_mesh,
            y_mat,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0)
        ).expect("Parent node must exist");

        // Create Z axis (blue)
        let z_mat = scene.add_material(
            Material::new().with_line_color(RgbaColor { r: 0.0, g: 0.0, b: 1.0, a: 1.0 }),
        );
        let z_start = origin;
        let z_end = origin + Vector3::new(0.0, 0.0, size);
        let z_vertices = vec![
            Vertex {
                position: z_start.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 0.0, 1.0]
            },
            Vertex {
                position: z_end.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 0.0, 1.0]
            },
        ];
        let z_primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices: vec![0, 1]
        }];
        let z_mesh = scene.add_mesh(Mesh::from_raw(z_vertices, z_primitives));
        scene.add_instance_node(
            Some(parent_node_id),
            z_mesh,
            z_mat,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0)
        ).expect("Parent node must exist");

        // Track annotation
        let annotation_id = self.next_annotation_id;
        self.next_annotation_id += 1;
        self.annotation_nodes.insert(annotation_id, parent_node_id);

        annotation_id
    }

    /// Adds one or more points at the specified positions.
    pub fn add_points(
        &mut self,
        scene: &mut Scene,
        positions: &[Point3<f32>],
        color: RgbaColor,
    ) -> AnnotationId {
        // Get annotation root node first (before other borrows)
        let root_id = self.root_node(scene);

        let vertices: Vec<Vertex> = positions
            .iter()
            .map(|&p| Vertex {
                position: p.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
            })
            .collect();

        let indices: Vec<u16> = (0..positions.len() as u16).collect();

        let primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::PointList,
            indices,
        }];

        // Create mesh
        let mesh = Mesh::from_raw(vertices, primitives);
        let mesh_id = scene.add_mesh(mesh);

        // Create material
        let material = Material::new().with_point_color(color);
        let material_id = scene.add_material(material);

        // Create node
        let node_id = scene.add_instance_node(
            Some(root_id),
            mesh_id,
            material_id,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        ).expect("Annotation root node must exist");

        // Track annotation
        let annotation_id = self.next_annotation_id;
        self.next_annotation_id += 1;
        self.annotation_nodes.insert(annotation_id, node_id);

        annotation_id
    }

    /// Adds a wireframe box.
    pub fn add_box(
        &mut self,
        scene: &mut Scene,
        center: Point3<f32>,
        size: Vector3<f32>,
        color: RgbaColor,
    ) -> AnnotationId {
        // Get annotation root node first (before other borrows)
        let root_id = self.root_node(scene);

        let half = size / 2.0;

        // Create 8 corners of the box
        let corners = [
            center + Vector3::new(-half.x, -half.y, -half.z), // 0: min corner
            center + Vector3::new( half.x, -half.y, -half.z), // 1
            center + Vector3::new( half.x,  half.y, -half.z), // 2
            center + Vector3::new(-half.x,  half.y, -half.z), // 3
            center + Vector3::new(-half.x, -half.y,  half.z), // 4
            center + Vector3::new( half.x, -half.y,  half.z), // 5
            center + Vector3::new( half.x,  half.y,  half.z), // 6: max corner
            center + Vector3::new(-half.x,  half.y,  half.z), // 7
        ];

        let vertices: Vec<Vertex> = corners
            .iter()
            .map(|&p| Vertex {
                position: p.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
            })
            .collect();

        // Create edges (12 edges for a box)
        let indices = vec![
            // Bottom face
            0, 1, 1, 2, 2, 3, 3, 0,
            // Top face
            4, 5, 5, 6, 6, 7, 7, 4,
            // Vertical edges
            0, 4, 1, 5, 2, 6, 3, 7,
        ];

        let primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices,
        }];

        // Create mesh
        let mesh = Mesh::from_raw(vertices, primitives);
        let mesh_id = scene.add_mesh(mesh);

        // Create material
        let material = Material::new().with_line_color(color);
        let material_id = scene.add_material(material);

        // Create node
        let node_id = scene.add_instance_node(
            Some(root_id),
            mesh_id,
            material_id,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        ).expect("Annotation root node must exist");

        // Track annotation
        let annotation_id = self.next_annotation_id;
        self.next_annotation_id += 1;
        self.annotation_nodes.insert(annotation_id, node_id);

        annotation_id
    }

    /// Adds a grid in the XZ plane.
    pub fn add_grid(
        &mut self,
        scene: &mut Scene,
        center: Point3<f32>,
        size: f32,
        divisions: u32,
        color: RgbaColor,
    ) -> AnnotationId {
        // Get annotation root node first (before other borrows)
        let root_id = self.root_node(scene);

        let half_size = size / 2.0;
        let step = size / divisions as f32;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Create grid lines parallel to X axis
        for i in 0..=divisions {
            let z = -half_size + i as f32 * step;
            let start = center + Vector3::new(-half_size, 0.0, z);
            let end = center + Vector3::new(half_size, 0.0, z);

            let idx = vertices.len() as u16;
            vertices.push(Vertex {
                position: start.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
            });
            vertices.push(Vertex {
                position: end.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
            });
            indices.push(idx);
            indices.push(idx + 1);
        }

        // Create grid lines parallel to Z axis
        for i in 0..=divisions {
            let x = -half_size + i as f32 * step;
            let start = center + Vector3::new(x, 0.0, -half_size);
            let end = center + Vector3::new(x, 0.0, half_size);

            let idx = vertices.len() as u16;
            vertices.push(Vertex {
                position: start.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
            });
            vertices.push(Vertex {
                position: end.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
            });
            indices.push(idx);
            indices.push(idx + 1);
        }

        let primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices,
        }];

        // Create mesh
        let mesh = Mesh::from_raw(vertices, primitives);
        let mesh_id = scene.add_mesh(mesh);

        // Create material
        let material = Material::new().with_line_color(color);
        let material_id = scene.add_material(material);

        // Create node
        let node_id = scene.add_instance_node(
            Some(root_id),
            mesh_id,
            material_id,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        ).expect("Annotation root node must exist");

        // Track annotation
        let annotation_id = self.next_annotation_id;
        self.next_annotation_id += 1;
        self.annotation_nodes.insert(annotation_id, node_id);

        annotation_id
    }
}
