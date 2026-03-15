use cgmath::Point3;

use crate::common::{ConvexPolyhedron, Ray};
use crate::Mesh;

/// Result of a ray-mesh intersection test in local mesh space.
#[derive(Debug, Clone)]
pub struct MeshHit {
    /// Distance along the ray to the hit point (in local space)
    pub distance: f32,
    /// Hit location in local mesh space
    pub hit_point: Point3<f32>,
    /// Index of the triangle that was hit (index into the mesh's index buffer / 3)
    pub triangle_index: usize,
    /// Barycentric coordinates of the hit point on the triangle (u, v, w) where w = 1 - u - v
    pub barycentric: (f32, f32, f32),
}

/// Result of a volume-mesh intersection test in local mesh space.
#[derive(Debug, Clone)]
pub struct MeshVolumeHit {
    /// Indices of triangles that intersect the volume
    pub triangle_indices: Vec<usize>,
    /// True if all triangles in the mesh are fully contained within the volume
    pub fully_contained: bool,
}

/// Tests a ray against all triangles in a mesh.
///
/// The ray should be in local mesh space. Returns all intersections found,
/// unsorted (caller can sort by distance if needed).
pub fn intersect_ray(mesh: &Mesh, ray: &Ray) -> Vec<MeshHit> {
    let mut hits = Vec::new();

    let triangle_indices = mesh.triangle_indices();

    for triangle_index in 0..(triangle_indices.len() / 3) {
        let i0 = triangle_indices[triangle_index * 3] as usize;
        let i1 = triangle_indices[triangle_index * 3 + 1] as usize;
        let i2 = triangle_indices[triangle_index * 3 + 2] as usize;

        let v0 = Point3::from(mesh.vertices()[i0].position);
        let v1 = Point3::from(mesh.vertices()[i1].position);
        let v2 = Point3::from(mesh.vertices()[i2].position);

        if let Some((t, u, v)) = ray.intersect_triangle(v0, v1, v2) {
            let w = 1.0 - u - v;
            hits.push(MeshHit {
                distance: t,
                hit_point: ray.point_at(t),
                triangle_index,
                barycentric: (u, v, w),
            });
        }
    }

    hits
}

/// Tests a convex volume against all triangles in a mesh.
///
/// The volume should be in local mesh space. Returns information about which
/// triangles intersect the volume and whether the entire mesh is contained.
///
/// # Arguments
/// * `mesh` - The mesh to test against
/// * `volume` - The convex polyhedron to test against (in local mesh space)
/// * `thorough` - If true, uses more accurate but slower edge-triangle intersection tests
///
/// # Returns
/// `Some(MeshVolumeHit)` if any triangles intersect the volume, `None` otherwise.
pub fn intersect_volume(
    mesh: &Mesh,
    volume: &ConvexPolyhedron,
    thorough: bool,
) -> Option<MeshVolumeHit> {
    let triangle_indices_data = mesh.triangle_indices();
    let num_triangles = triangle_indices_data.len() / 3;

    if num_triangles == 0 {
        return None;
    }

    let mut hit_indices = Vec::new();
    let mut all_fully_contained = true;

    for triangle_index in 0..num_triangles {
        let i0 = triangle_indices_data[triangle_index * 3] as usize;
        let i1 = triangle_indices_data[triangle_index * 3 + 1] as usize;
        let i2 = triangle_indices_data[triangle_index * 3 + 2] as usize;

        let v0 = Point3::from(mesh.vertices()[i0].position);
        let v1 = Point3::from(mesh.vertices()[i1].position);
        let v2 = Point3::from(mesh.vertices()[i2].position);

        let fully_inside = volume.contains_triangle(v0, v1, v2);

        if fully_inside {
            hit_indices.push(triangle_index);
        } else if volume.intersects_triangle(v0, v1, v2, thorough) {
            hit_indices.push(triangle_index);
            all_fully_contained = false;
        } else {
            all_fully_contained = false;
        }
    }

    if hit_indices.is_empty() {
        None
    } else {
        Some(MeshVolumeHit {
            triangle_indices: hit_indices,
            fully_contained: all_fully_contained,
        })
    }
}
