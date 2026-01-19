use std::collections::HashMap;

use cgmath::{Point3, Vector3};

use crate::{
    common::RgbaColor,
    scene::{Material, Mesh, MeshPrimitive, NodeId, PrimitiveType, Vertex},
};

/// Unique identifier for annotations
pub type AnnotationId = u32;

/// Data needed to create scene geometry for an annotation
pub struct AnnotationMeshData {
    pub mesh: Mesh,
    pub material: Material,
    pub name: Option<String>,
}

/// Common metadata for all annotations
#[derive(Debug, Clone)]
pub struct AnnotationMeta {
    /// Unique ID assigned when added to manager
    pub id: AnnotationId,
    /// Optional user-provided name for debugging
    pub name: Option<String>,
    /// Whether this annotation is currently visible
    pub visible: bool,
    /// Node ID if this annotation has been reified (rendered to scene graph)
    pub node_id: Option<NodeId>,
}

impl AnnotationMeta {
    fn new(id: AnnotationId) -> Self {
        Self {
            id,
            name: None,
            visible: true,
            node_id: None,
        }
    }
}

/// A single line segment annotation
#[derive(Debug, Clone)]
pub struct LineAnnotation {
    pub meta: AnnotationMeta,
    pub start: Point3<f32>,
    pub end: Point3<f32>,
    pub color: RgbaColor,
}

impl LineAnnotation {
    /// Creates mesh data for this line annotation
    pub fn to_mesh_data(&self) -> AnnotationMeshData {
        let vertices = vec![
            Vertex {
                position: self.start.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
            },
            Vertex {
                position: self.end.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
            },
        ];

        let primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices: vec![0, 1],
        }];

        AnnotationMeshData {
            mesh: Mesh::from_raw(vertices, primitives),
            material: Material::new().with_line_color(self.color),
            name: self.meta.name.clone(),
        }
    }
}

/// A connected series of line segments
#[derive(Debug, Clone)]
pub struct PolylineAnnotation {
    pub meta: AnnotationMeta,
    pub points: Vec<Point3<f32>>,
    pub color: RgbaColor,
    /// If true, connect last point back to first
    pub closed: bool,
}

impl PolylineAnnotation {
    /// Creates mesh data for this polyline annotation.
    /// Returns None if points is empty.
    pub fn to_mesh_data(&self) -> Option<AnnotationMeshData> {
        if self.points.is_empty() {
            return None;
        }

        let vertices: Vec<Vertex> = self
            .points
            .iter()
            .map(|&p| Vertex {
                position: p.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
            })
            .collect();

        let mut indices = Vec::new();
        for i in 0..self.points.len().saturating_sub(1) {
            indices.push(i as u16);
            indices.push((i + 1) as u16);
        }
        if self.closed && self.points.len() > 2 {
            indices.push((self.points.len() - 1) as u16);
            indices.push(0);
        }

        let primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices,
        }];

        Some(AnnotationMeshData {
            mesh: Mesh::from_raw(vertices, primitives),
            material: Material::new().with_line_color(self.color),
            name: self.meta.name.clone(),
        })
    }
}

/// A set of point markers
#[derive(Debug, Clone)]
pub struct PointsAnnotation {
    pub meta: AnnotationMeta,
    pub positions: Vec<Point3<f32>>,
    pub color: RgbaColor,
}

impl PointsAnnotation {
    /// Creates mesh data for this points annotation.
    /// Returns None if positions is empty.
    pub fn to_mesh_data(&self) -> Option<AnnotationMeshData> {
        if self.positions.is_empty() {
            return None;
        }

        let vertices: Vec<Vertex> = self
            .positions
            .iter()
            .map(|&p| Vertex {
                position: p.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
            })
            .collect();

        let indices: Vec<u16> = (0..self.positions.len() as u16).collect();

        let primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::PointList,
            indices,
        }];

        Some(AnnotationMeshData {
            mesh: Mesh::from_raw(vertices, primitives),
            material: Material::new().with_point_color(self.color),
            name: self.meta.name.clone(),
        })
    }
}

/// Coordinate axes (RGB = XYZ)
#[derive(Debug, Clone)]
pub struct AxesAnnotation {
    pub meta: AnnotationMeta,
    pub origin: Point3<f32>,
    pub size: f32,
}

impl AxesAnnotation {
    /// Creates mesh data for each axis (X=red, Y=green, Z=blue).
    /// Returns a Vec with 3 mesh data items, one per axis.
    pub fn to_mesh_data(&self) -> Vec<AnnotationMeshData> {
        let axes = [
            (Vector3::new(self.size, 0.0, 0.0), RgbaColor::RED),
            (Vector3::new(0.0, self.size, 0.0), RgbaColor::GREEN),
            (Vector3::new(0.0, 0.0, self.size), RgbaColor::BLUE),
        ];

        axes.iter()
            .map(|(dir, color)| {
                let end = self.origin + dir;
                let vertices = vec![
                    Vertex {
                        position: self.origin.into(),
                        tex_coords: [0.0; 3],
                        normal: [0.0, 1.0, 0.0],
                    },
                    Vertex {
                        position: end.into(),
                        tex_coords: [0.0; 3],
                        normal: [0.0, 1.0, 0.0],
                    },
                ];

                let primitives = vec![MeshPrimitive {
                    primitive_type: PrimitiveType::LineList,
                    indices: vec![0, 1],
                }];

                AnnotationMeshData {
                    mesh: Mesh::from_raw(vertices, primitives),
                    material: Material::new().with_line_color(*color),
                    name: None,
                }
            })
            .collect()
    }
}

/// A wireframe box
#[derive(Debug, Clone)]
pub struct BoxAnnotation {
    pub meta: AnnotationMeta,
    pub center: Point3<f32>,
    pub size: Vector3<f32>,
    pub color: RgbaColor,
}

impl BoxAnnotation {
    /// Creates mesh data for this wireframe box annotation
    pub fn to_mesh_data(&self) -> AnnotationMeshData {
        let half = self.size / 2.0;

        let corners = [
            self.center + Vector3::new(-half.x, -half.y, -half.z),
            self.center + Vector3::new(half.x, -half.y, -half.z),
            self.center + Vector3::new(half.x, half.y, -half.z),
            self.center + Vector3::new(-half.x, half.y, -half.z),
            self.center + Vector3::new(-half.x, -half.y, half.z),
            self.center + Vector3::new(half.x, -half.y, half.z),
            self.center + Vector3::new(half.x, half.y, half.z),
            self.center + Vector3::new(-half.x, half.y, half.z),
        ];

        let vertices: Vec<Vertex> = corners
            .iter()
            .map(|&p| Vertex {
                position: p.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
            })
            .collect();

        // 12 edges of a box
        let indices = vec![
            0, 1, 1, 2, 2, 3, 3, 0, // Bottom face
            4, 5, 5, 6, 6, 7, 7, 4, // Top face
            0, 4, 1, 5, 2, 6, 3, 7, // Vertical edges
        ];

        let primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices,
        }];

        AnnotationMeshData {
            mesh: Mesh::from_raw(vertices, primitives),
            material: Material::new().with_line_color(self.color),
            name: self.meta.name.clone(),
        }
    }
}

/// A grid in the XZ plane
#[derive(Debug, Clone)]
pub struct GridAnnotation {
    pub meta: AnnotationMeta,
    pub center: Point3<f32>,
    pub size: f32,
    pub divisions: u32,
    pub color: RgbaColor,
}

impl GridAnnotation {
    /// Creates mesh data for this grid annotation
    pub fn to_mesh_data(&self) -> AnnotationMeshData {
        let half_size = self.size / 2.0;
        let step = self.size / self.divisions as f32;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Lines parallel to X axis
        for i in 0..=self.divisions {
            let z = -half_size + i as f32 * step;
            let start = self.center + Vector3::new(-half_size, 0.0, z);
            let end = self.center + Vector3::new(half_size, 0.0, z);

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

        // Lines parallel to Z axis
        for i in 0..=self.divisions {
            let x = -half_size + i as f32 * step;
            let start = self.center + Vector3::new(x, 0.0, -half_size);
            let end = self.center + Vector3::new(x, 0.0, half_size);

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

        AnnotationMeshData {
            mesh: Mesh::from_raw(vertices, primitives),
            material: Material::new().with_line_color(self.color),
            name: self.meta.name.clone(),
        }
    }
}

/// Enum encompassing all annotation types
#[derive(Debug, Clone)]
pub enum Annotation {
    Line(LineAnnotation),
    Polyline(PolylineAnnotation),
    Points(PointsAnnotation),
    Axes(AxesAnnotation),
    Box(BoxAnnotation),
    Grid(GridAnnotation),
}

impl Annotation {
    /// Get the annotation's metadata
    pub fn meta(&self) -> &AnnotationMeta {
        match self {
            Annotation::Line(a) => &a.meta,
            Annotation::Polyline(a) => &a.meta,
            Annotation::Points(a) => &a.meta,
            Annotation::Axes(a) => &a.meta,
            Annotation::Box(a) => &a.meta,
            Annotation::Grid(a) => &a.meta,
        }
    }

    /// Get mutable reference to annotation's metadata
    pub fn meta_mut(&mut self) -> &mut AnnotationMeta {
        match self {
            Annotation::Line(a) => &mut a.meta,
            Annotation::Polyline(a) => &mut a.meta,
            Annotation::Points(a) => &mut a.meta,
            Annotation::Axes(a) => &mut a.meta,
            Annotation::Box(a) => &mut a.meta,
            Annotation::Grid(a) => &mut a.meta,
        }
    }

    /// Returns the annotation ID
    pub fn id(&self) -> AnnotationId {
        self.meta().id
    }

    /// Returns whether this annotation has been reified to the scene graph
    pub fn is_reified(&self) -> bool {
        self.meta().node_id.is_some()
    }

    /// Returns the node ID if reified
    pub fn node_id(&self) -> Option<NodeId> {
        self.meta().node_id
    }
}

/// Manages 3D annotations with lazy reification.
///
/// Annotations are stored as data objects and can be reified (converted to
/// scene nodes with meshes and materials) on demand. This separation allows
/// annotations to survive scene clearing operations if desired, or to be
/// automatically cleared with the scene.
///
/// The AnnotationManager is owned by Scene, ensuring lifecycle consistency.
pub struct AnnotationManager {
    /// All annotations indexed by their ID
    annotations: HashMap<AnnotationId, Annotation>,

    /// Next annotation ID to assign
    next_id: AnnotationId,

    /// Root node for all annotation geometry (lazy initialized)
    root_node: Option<NodeId>,
}

impl Default for AnnotationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AnnotationManager {
    /// Creates a new empty annotation manager
    pub fn new() -> Self {
        Self {
            annotations: HashMap::new(),
            next_id: 0,
            root_node: None,
        }
    }

    /// Returns the number of annotations
    pub fn len(&self) -> usize {
        self.annotations.len()
    }

    /// Returns true if there are no annotations
    pub fn is_empty(&self) -> bool {
        self.annotations.is_empty()
    }

    /// Returns the number of unreified annotations
    pub fn unreified_count(&self) -> usize {
        self.annotations.values().filter(|a| !a.is_reified()).count()
    }

    /// Get an annotation by ID
    pub fn get(&self, id: AnnotationId) -> Option<&Annotation> {
        self.annotations.get(&id)
    }

    /// Get a mutable annotation by ID
    pub fn get_mut(&mut self, id: AnnotationId) -> Option<&mut Annotation> {
        self.annotations.get_mut(&id)
    }

    /// Iterate over all annotations
    pub fn iter(&self) -> impl Iterator<Item = &Annotation> {
        self.annotations.values()
    }

    /// Iterate over unreified annotations
    pub fn iter_unreified(&self) -> impl Iterator<Item = &Annotation> {
        self.annotations.values().filter(|a| !a.is_reified())
    }

    /// Clears all annotations and resets state.
    pub fn clear(&mut self) {
        self.annotations.clear();
        self.next_id = 0;
        self.root_node = None;
    }

    /// Marks all annotations as unreified (removes node_id references).
    /// Called when nodes are cleared but annotations should be retained.
    pub fn mark_all_unreified(&mut self) {
        for annotation in self.annotations.values_mut() {
            annotation.meta_mut().node_id = None;
        }
        self.root_node = None;
    }

    /// Allocate a new annotation ID
    fn next_id(&mut self) -> AnnotationId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    // ========== Add annotation methods ==========
    // These add annotation data but do NOT create scene nodes

    /// Adds a line annotation (data only, not yet reified)
    pub fn add_line(
        &mut self,
        start: Point3<f32>,
        end: Point3<f32>,
        color: RgbaColor,
    ) -> AnnotationId {
        let id = self.next_id();
        let annotation = Annotation::Line(LineAnnotation {
            meta: AnnotationMeta::new(id),
            start,
            end,
            color,
        });
        self.annotations.insert(id, annotation);
        id
    }

    /// Adds a polyline annotation (data only, not yet reified)
    pub fn add_polyline(
        &mut self,
        points: Vec<Point3<f32>>,
        color: RgbaColor,
        closed: bool,
    ) -> AnnotationId {
        let id = self.next_id();
        let annotation = Annotation::Polyline(PolylineAnnotation {
            meta: AnnotationMeta::new(id),
            points,
            color,
            closed,
        });
        self.annotations.insert(id, annotation);
        id
    }

    /// Adds a points annotation (data only, not yet reified)
    pub fn add_points(&mut self, positions: Vec<Point3<f32>>, color: RgbaColor) -> AnnotationId {
        let id = self.next_id();
        let annotation = Annotation::Points(PointsAnnotation {
            meta: AnnotationMeta::new(id),
            positions,
            color,
        });
        self.annotations.insert(id, annotation);
        id
    }

    /// Adds an axes annotation (data only, not yet reified)
    pub fn add_axes(&mut self, origin: Point3<f32>, size: f32) -> AnnotationId {
        let id = self.next_id();
        let annotation = Annotation::Axes(AxesAnnotation {
            meta: AnnotationMeta::new(id),
            origin,
            size,
        });
        self.annotations.insert(id, annotation);
        id
    }

    /// Adds a box annotation (data only, not yet reified)
    pub fn add_box(
        &mut self,
        center: Point3<f32>,
        size: Vector3<f32>,
        color: RgbaColor,
    ) -> AnnotationId {
        let id = self.next_id();
        let annotation = Annotation::Box(BoxAnnotation {
            meta: AnnotationMeta::new(id),
            center,
            size,
            color,
        });
        self.annotations.insert(id, annotation);
        id
    }

    /// Adds a grid annotation (data only, not yet reified)
    pub fn add_grid(
        &mut self,
        center: Point3<f32>,
        size: f32,
        divisions: u32,
        color: RgbaColor,
    ) -> AnnotationId {
        let id = self.next_id();
        let annotation = Annotation::Grid(GridAnnotation {
            meta: AnnotationMeta::new(id),
            center,
            size,
            divisions,
            color,
        });
        self.annotations.insert(id, annotation);
        id
    }

    /// Removes an annotation by ID.
    /// Returns the removed annotation if found.
    /// Note: This does NOT remove the scene node - use Scene::remove_annotation() instead.
    pub(crate) fn remove(&mut self, id: AnnotationId) -> Option<Annotation> {
        self.annotations.remove(&id)
    }

    /// Sets the root node ID (called during reification)
    pub(crate) fn set_root_node(&mut self, node_id: NodeId) {
        self.root_node = Some(node_id);
    }

    /// Gets the root node ID
    pub fn root_node(&self) -> Option<NodeId> {
        self.root_node
    }
}
