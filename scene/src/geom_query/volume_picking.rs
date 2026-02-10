use cgmath::Matrix4;

use crate::common::{Aabb, ConvexPolyhedron};
use crate::{InstanceId, Mesh, NodeId, Scene};

use super::pick_query::{pick_all, PickQuery};

/// Result of a volume-instance intersection test.
#[derive(Debug, Clone)]
pub struct VolumePickResult {
    /// The node that was hit
    pub node_id: NodeId,
    /// The instance that was hit
    pub instance_id: InstanceId,
    /// Indices of triangles that intersect the volume
    pub triangle_indices: Vec<usize>,
    /// True if the entire instance is fully contained within the volume
    pub fully_contained: bool,
}

/// Volume picking query that implements the generic PickQuery trait.
///
/// Wraps a ConvexPolyhedron with a flag for thorough testing.
pub struct VolumePickQuery {
    /// The volume in current coordinate space (may be transformed to local space)
    volume: ConvexPolyhedron,
    /// Whether to use thorough (but slower) edge-triangle intersection tests
    thorough: bool,
}

impl VolumePickQuery {
    /// Creates a new volume pick query.
    ///
    /// # Arguments
    /// * `volume` - The convex polyhedron to test against (in world space)
    /// * `thorough` - If true, uses more accurate but slower edge-triangle tests
    pub fn new(volume: ConvexPolyhedron, thorough: bool) -> Self {
        Self { volume, thorough }
    }
}

impl PickQuery for VolumePickQuery {
    type Result = VolumePickResult;

    fn might_intersect_bounds(&self, bounds: &Aabb) -> bool {
        self.volume.intersects_aabb(bounds)
    }

    fn transform(&self, matrix: &Matrix4<f32>) -> Self {
        Self {
            volume: self.volume.transform(matrix),
            thorough: self.thorough,
        }
    }

    fn collect_mesh_hits(
        &self,
        mesh: &Mesh,
        node_id: NodeId,
        instance_id: InstanceId,
        _world_transform: &Matrix4<f32>,
        results: &mut Vec<Self::Result>,
    ) {
        // Test against mesh (volume is already in local space)
        if let Some(mesh_hit) = mesh.intersect_volume(&self.volume, self.thorough) {
            results.push(VolumePickResult {
                node_id,
                instance_id,
                triangle_indices: mesh_hit.triangle_indices,
                fully_contained: mesh_hit.fully_contained,
            });
        }
    }
}

/// Picks all instances intersected by a convex volume.
///
/// The volume should be in world space. The function walks the scene tree from root nodes,
/// using cached bounding boxes to eliminate large portions of the scene efficiently.
///
/// # Arguments
/// * `volume` - The convex polyhedron to test against (in world space)
/// * `scene` - The scene to pick from
/// * `thorough` - If true, uses more accurate but slower edge-triangle intersection tests.
///   This catches edge cases where the volume passes through a triangle without any triangle
///   vertices being inside and without triangle edges crossing the volume boundary.
///
/// # Returns
/// A vector of VolumePickResult for each instance that intersects the volume.
/// Each result includes whether the instance is fully contained within the volume.
pub fn pick_all_from_volume(
    volume: &ConvexPolyhedron,
    scene: &Scene,
    thorough: bool,
) -> Vec<VolumePickResult> {
    let query = VolumePickQuery::new(volume.clone(), thorough);
    pick_all(&query, scene)
}
