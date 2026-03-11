use std::collections::HashMap;

use cgmath::{EuclideanSpace, Point3, Vector3};

use super::{Material, Mesh, MeshPrimitive, NodeId, PrimitiveType, Vertex};
use crate::common::RgbaColor;

/// Unique identifier for annotations
pub type AnnotationId = u32;

/// Data needed to create scene geometry for an annotation
pub(crate) struct AnnotationMeshData {
    pub mesh: Mesh,
    pub material: Material,
    pub name: Option<String>,
}

/// Common metadata for all annotations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AnnotationMeta {
    /// Unique ID assigned when added to manager
    pub id: AnnotationId,
    /// Optional user-provided name for debugging
    pub name: Option<String>,
    /// Whether this annotation is currently visible
    pub visible: bool,
    /// Node ID if this annotation has been reified
    #[cfg_attr(feature = "serde", serde(skip))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LineAnnotation {
    pub meta: AnnotationMeta,
    pub start: Point3<f32>,
    pub end: Point3<f32>,
    pub color: RgbaColor,
}

impl LineAnnotation {
    /// Creates mesh data for this line annotation
    pub(crate) fn to_mesh_data(&self) -> AnnotationMeshData {
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    pub(crate) fn to_mesh_data(&self) -> Option<AnnotationMeshData> {
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointsAnnotation {
    pub meta: AnnotationMeta,
    pub positions: Vec<Point3<f32>>,
    pub color: RgbaColor,
}

impl PointsAnnotation {
    /// Creates mesh data for this points annotation.
    /// Returns None if positions is empty.
    pub(crate) fn to_mesh_data(&self) -> Option<AnnotationMeshData> {
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AxesAnnotation {
    pub meta: AnnotationMeta,
    pub origin: Point3<f32>,
    pub size: f32,
}

impl AxesAnnotation {
    /// Creates mesh data for each axis (X=red, Y=green, Z=blue).
    /// Returns a Vec with 3 mesh data items, one per axis.
    pub(crate) fn to_mesh_data(&self) -> Vec<AnnotationMeshData> {
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BoxAnnotation {
    pub meta: AnnotationMeta,
    pub center: Point3<f32>,
    pub size: Vector3<f32>,
    pub color: RgbaColor,
}

impl BoxAnnotation {
    /// Creates mesh data for this wireframe box annotation
    pub(crate) fn to_mesh_data(&self) -> AnnotationMeshData {
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GridAnnotation {
    pub meta: AnnotationMeta,
    pub center: Point3<f32>,
    pub size: f32,
    pub divisions: u32,
    pub color: RgbaColor,
}

impl GridAnnotation {
    /// Creates mesh data for this grid annotation
    pub(crate) fn to_mesh_data(&self) -> AnnotationMeshData {
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

/// Visualization of a point light as a wireframe sphere.
///
/// Stores a reference to a light by index. Geometry is built during reification
/// from the live light data using [`Mesh::sphere`] with [`PrimitiveType::LineList`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointLightAnnotation {
    pub meta: AnnotationMeta,
    /// Index into `scene.lights`
    pub light_index: usize,
    /// Radius of the indicator wireframe
    pub radius: f32,
    /// Number of segments for the wireframe sphere
    pub segments: u32,
    /// Generation of the light data when last reified
    pub reified_generation: Option<u64>,
}

impl PointLightAnnotation {
    /// Creates wireframe sphere geometry for the point light visualization.
    ///
    /// The sphere is centered at `position` using the light's `color`.
    pub(crate) fn to_mesh_data(&self, position: Point3<f32>, color: RgbaColor) -> AnnotationMeshData {
        let mesh = Mesh::sphere(self.radius, self.segments, self.segments / 2, PrimitiveType::LineList)
            .translated(position.to_vec());

        AnnotationMeshData {
            mesh,
            material: Material::new().with_line_color(color),
            name: self.meta.name.clone(),
        }
    }
}

/// Visualization of a spot light as a cone outline.
///
/// Stores a reference to a light by index. Geometry is built during reification
/// from the live light data using [`Mesh::cone_directed`] with [`PrimitiveType::LineList`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpotLightAnnotation {
    pub meta: AnnotationMeta,
    /// Index into `scene.lights`
    pub light_index: usize,
    /// Length of the cone visualization
    pub length: f32,
    /// Number of segments forming the cone circumference
    pub segments: u32,
    /// Generation of the light data when last reified
    pub reified_generation: Option<u64>,
}

impl SpotLightAnnotation {
    /// Creates cone outline geometry for the spot light visualization.
    ///
    /// Returns 2 meshes: outer cone (full color) and inner cone (dimmer).
    pub(crate) fn to_mesh_data(
        &self,
        position: Point3<f32>,
        direction: Vector3<f32>,
        color: RgbaColor,
        inner_cone_angle: f32,
        outer_cone_angle: f32,
    ) -> Vec<AnnotationMeshData> {
        let cones = [
            (outer_cone_angle, color, "SpotLight_Outer"),
            (
                inner_cone_angle,
                RgbaColor {
                    r: color.r * 0.5,
                    g: color.g * 0.5,
                    b: color.b * 0.5,
                    a: color.a,
                },
                "SpotLight_Inner",
            ),
        ];

        cones
            .iter()
            .map(|&(cone_angle, cone_color, name)| {
                let cone_radius = self.length * cone_angle.tan();
                let mesh = Mesh::cone_directed(
                    position,
                    direction,
                    cone_radius,
                    self.length,
                    self.segments,
                    false,
                    PrimitiveType::LineList,
                );

                AnnotationMeshData {
                    mesh,
                    material: Material::new().with_line_color(cone_color),
                    name: Some(name.to_string()),
                }
            })
            .collect()
    }
}

/// Visualization of vertex normals as short lines.
///
/// Stores a reference to a node by ID. Geometry is built during reification
/// by looking up the node's instance mesh and world transform.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NormalsAnnotation {
    pub meta: AnnotationMeta,
    /// Node whose instance's mesh normals to visualize
    pub target_node_id: NodeId,
    /// Color for the normal lines
    pub color: RgbaColor,
    /// Length of each normal line
    pub length: f32,
    /// Generation of the mesh data when last reified
    pub reified_generation: Option<u64>,
}

/// Maximum normals per mesh chunk (constrained by u16 index limit).
/// Each normal uses 2 vertices, so 32,000 vertices = 16,000 normals.
const MAX_NORMALS_PER_CHUNK: usize = 16_000;

impl NormalsAnnotation {
    /// Creates normal visualization geometry from world-space vertex data.
    ///
    /// Each entry in `vertices` is a (position, normal) pair in world space.
    /// Splits into multiple meshes if the vertex count exceeds u16 index limits.
    pub(crate) fn to_mesh_data(
        &self,
        vertices: &[(Point3<f32>, Vector3<f32>)],
    ) -> Vec<AnnotationMeshData> {
        if vertices.is_empty() {
            return Vec::new();
        }

        vertices
            .chunks(MAX_NORMALS_PER_CHUNK)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let mut mesh_vertices = Vec::with_capacity(chunk.len() * 2);
                let mut indices = Vec::with_capacity(chunk.len() * 2);

                for (i, (pos, normal)) in chunk.iter().enumerate() {
                    let end = pos + normal * self.length;
                    mesh_vertices.push(Vertex {
                        position: (*pos).into(),
                        tex_coords: [0.0; 3],
                        normal: [0.0, 1.0, 0.0],
                    });
                    mesh_vertices.push(Vertex {
                        position: end.into(),
                        tex_coords: [0.0; 3],
                        normal: [0.0, 1.0, 0.0],
                    });
                    indices.push((i * 2) as u16);
                    indices.push((i * 2 + 1) as u16);
                }

                let primitives = vec![MeshPrimitive {
                    primitive_type: PrimitiveType::LineList,
                    indices,
                }];

                let name = if chunk_idx == 0 {
                    self.meta
                        .name
                        .clone()
                        .or_else(|| Some("Normals".to_string()))
                } else {
                    Some(format!(
                        "{}_{}",
                        self.meta.name.as_deref().unwrap_or("Normals"),
                        chunk_idx
                    ))
                };

                AnnotationMeshData {
                    mesh: Mesh::from_raw(mesh_vertices, primitives),
                    material: Material::new().with_line_color(self.color),
                    name,
                }
            })
            .collect()
    }
}

/// Enum encompassing all annotation types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Annotation {
    Line(LineAnnotation),
    Polyline(PolylineAnnotation),
    Points(PointsAnnotation),
    Axes(AxesAnnotation),
    Box(BoxAnnotation),
    Grid(GridAnnotation),
    PointLight(PointLightAnnotation),
    SpotLight(SpotLightAnnotation),
    Normals(NormalsAnnotation),
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
            Annotation::PointLight(a) => &a.meta,
            Annotation::SpotLight(a) => &a.meta,
            Annotation::Normals(a) => &a.meta,
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
            Annotation::PointLight(a) => &mut a.meta,
            Annotation::SpotLight(a) => &mut a.meta,
            Annotation::Normals(a) => &mut a.meta,
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
/// scene nodes with meshes and materials) on demand.
///
/// The AnnotationManager is owned by Scene, ensuring lifecycle consistency.
#[derive(Clone, Debug)]
pub struct AnnotationManager {
    /// All annotations indexed by their ID
    annotations: HashMap<AnnotationId, Annotation>,

    /// Next annotation ID to assign
    next_id: AnnotationId,

    /// Root node for all annotation geometry (lazy initialized)
    root_node: Option<NodeId>,
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

    /// Adds a line annotation
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

    /// Adds a polyline annotation
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

    /// Adds a points annotation
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

    /// Adds an axes annotation
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

    /// Adds a box annotation
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

    /// Adds a grid annotation
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

    /// Adds a point light annotation.
    ///
    /// Geometry is built during reification from the light at `light_index`.
    pub fn add_point_light(
        &mut self,
        light_index: usize,
        radius: f32,
        segments: u32,
    ) -> AnnotationId {
        let id = self.next_id();
        let annotation = Annotation::PointLight(PointLightAnnotation {
            meta: AnnotationMeta::new(id),
            light_index,
            radius,
            segments,
            reified_generation: None,
        });
        self.annotations.insert(id, annotation);
        id
    }

    /// Adds a spot light annotation.
    ///
    /// Geometry is built during reification from the light at `light_index`.
    pub fn add_spot_light(
        &mut self,
        light_index: usize,
        length: f32,
        segments: u32,
    ) -> AnnotationId {
        let id = self.next_id();
        let annotation = Annotation::SpotLight(SpotLightAnnotation {
            meta: AnnotationMeta::new(id),
            light_index,
            length,
            segments,
            reified_generation: None,
        });
        self.annotations.insert(id, annotation);
        id
    }

    /// Adds a normals annotation.
    ///
    /// Geometry is built during reification from the mesh of the node at `node_id`.
    pub fn add_normals(
        &mut self,
        node_id: NodeId,
        color: RgbaColor,
        length: f32,
    ) -> AnnotationId {
        let id = self.next_id();
        let annotation = Annotation::Normals(NormalsAnnotation {
            meta: AnnotationMeta::new(id),
            target_node_id: node_id,
            color,
            length,
            reified_generation: None,
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

    /// Inserts an annotation with a specific ID (used during deserialization).
    /// Updates next_id if needed to avoid ID collisions.
    /// Returns an error if an annotation with the same ID already exists.
    pub fn insert_with_id(&mut self, annotation: Annotation) -> anyhow::Result<()> {
        let id = annotation.id();
        if self.annotations.contains_key(&id) {
            anyhow::bail!("Annotation with ID {} already exists", id);
        }
        self.annotations.insert(id, annotation);
        // Ensure next_id is greater than any inserted ID
        if id >= self.next_id {
            self.next_id = id + 1;
        }
        Ok(())
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
