use cgmath::Point3;

use crate::common::{ConvexPolyhedron, Ray};
use crate::Mesh;

/// Result of a ray-triangle intersection test in local mesh space.
#[derive(Debug, Clone)]
pub struct TriangleMeshHit {
    /// Distance along the ray to the hit point (in local space)
    pub distance: f32,
    /// Hit location in local mesh space
    pub hit_point: Point3<f32>,
    /// Index of the triangle that was hit (index into the mesh's index buffer / 3)
    pub triangle_index: usize,
    /// Barycentric coordinates of the hit point on the triangle (u, v, w) where w = 1 - u - v
    pub barycentric: (f32, f32, f32),
}

/// Result of a ray-segment closest-approach test in local mesh space.
#[derive(Debug, Clone)]
pub struct LineMeshHit {
    /// Parameter along the ray at the closest approach point (in local space)
    pub t: f32,
    /// Closest point on the segment to the ray (in local mesh space)
    pub closest_point: Point3<f32>,
    /// Minimum distance between the ray and the segment
    pub distance_to_ray: f32,
    /// Index of the segment (0-based pair index into the mesh's line index buffer)
    pub segment_index: usize,
}

/// Result of a volume-mesh intersection test in local mesh space.
#[derive(Debug, Clone)]
pub struct MeshVolumeHit {
    /// Indices of triangles that intersect the volume
    pub triangle_indices: Vec<usize>,
    /// Indices of line segments that intersect the volume
    pub segment_indices: Vec<usize>,
    /// True if all geometry (triangles and line segments) in the mesh is fully contained within the volume
    pub fully_contained: bool,
}

/// Tests a ray against all triangles in a mesh.
///
/// The ray should be in local mesh space. Returns all intersections found,
/// unsorted (caller can sort by distance if needed).
pub fn intersect_ray(mesh: &Mesh, ray: &Ray) -> Vec<TriangleMeshHit> {
    let mut hits = Vec::new();

    for (triangle_index, [v0, v1, v2]) in mesh.triangles().enumerate() {
        let p0 = Point3::from(v0.position);
        let p1 = Point3::from(v1.position);
        let p2 = Point3::from(v2.position);

        if let Some((t, u, v)) = ray.intersect_triangle(p0, p1, p2) {
            let w = 1.0 - u - v;
            hits.push(TriangleMeshHit {
                distance: t,
                hit_point: ray.point_at(t),
                triangle_index,
                barycentric: (u, v, w),
            });
        }
    }

    hits
}

/// Tests a ray against all line segments in a mesh using closest-approach distance.
///
/// The ray should be in local mesh space. Returns all segments whose closest approach
/// to the ray is within `tolerance`. Results are unsorted (caller sorts if needed).
pub fn intersect_ray_with_lines(mesh: &Mesh, ray: &Ray, tolerance: f32) -> Vec<LineMeshHit> {
    let mut hits = Vec::new();

    for (segment_index, [v0, v1]) in mesh.segments().enumerate() {
        let p0 = Point3::from(v0.position);
        let p1 = Point3::from(v1.position);

        let Some(approach) = ray.closest_approach_to_segment(p0, p1) else {
            continue;
        };

        if approach.distance <= tolerance {
            hits.push(LineMeshHit {
                t: approach.t,
                closest_point: approach.closest_on_segment,
                distance_to_ray: approach.distance,
                segment_index,
            });
        }
    }

    hits
}

/// Tests a convex volume against all triangles and line segments in a mesh.
///
/// The volume should be in local mesh space. Returns information about which
/// triangles and segments intersect the volume and whether the entire mesh is contained.
///
/// # Arguments
/// * `mesh` - The mesh to test against
/// * `volume` - The convex polyhedron to test against (in local mesh space)
/// * `thorough` - If true, uses more accurate but slower edge-triangle intersection tests
///
/// # Returns
/// `Some(MeshVolumeHit)` if any geometry intersects the volume, `None` otherwise.
pub fn intersect_volume(
    mesh: &Mesh,
    volume: &ConvexPolyhedron,
    thorough: bool,
) -> Option<MeshVolumeHit> {
    let mut triangle_indices = Vec::new();
    let mut all_fully_contained = true;

    for (triangle_index, [v0, v1, v2]) in mesh.triangles().enumerate() {
        let p0 = Point3::from(v0.position);
        let p1 = Point3::from(v1.position);
        let p2 = Point3::from(v2.position);

        let fully_inside = volume.contains_triangle(p0, p1, p2);

        if fully_inside {
            triangle_indices.push(triangle_index);
        } else if volume.intersects_triangle(p0, p1, p2, thorough) {
            triangle_indices.push(triangle_index);
            all_fully_contained = false;
        } else {
            all_fully_contained = false;
        }
    }

    let mut segment_indices = Vec::new();

    for (segment_index, [v0, v1]) in mesh.segments().enumerate() {
        let p0 = Point3::from(v0.position);
        let p1 = Point3::from(v1.position);

        let p0_inside = volume.contains_point(p0);
        let p1_inside = volume.contains_point(p1);

        if p0_inside && p1_inside {
            segment_indices.push(segment_index);
        } else {
            let mid = p0 + (p1 - p0) * 0.5;
            if volume.contains_point(mid) {
                segment_indices.push(segment_index);
            }
            all_fully_contained = false;
        }
    }

    if triangle_indices.is_empty() && segment_indices.is_empty() {
        None
    } else {
        Some(MeshVolumeHit {
            triangle_indices,
            segment_indices,
            fully_contained: all_fully_contained,
        })
    }
}

#[cfg(test)]
mod tests {
    use cgmath::{Point3, Vector3};

    use crate::common::{Aabb, ConvexPolyhedron, Ray};
    use crate::{Mesh, MeshPrimitive, PrimitiveType, Vertex};

    use super::{intersect_ray_with_lines, intersect_volume};

    fn make_vertex(x: f32, y: f32, z: f32) -> Vertex {
        Vertex { position: [x, y, z], tex_coords: [0.0; 3], normal: [0.0, 1.0, 0.0] }
    }

    fn make_line_mesh(segments: &[(usize, usize)], vertices: Vec<Vertex>) -> Mesh {
        let indices: Vec<u32> = segments
            .iter()
            .flat_map(|&(a, b)| [a as u32, b as u32])
            .collect();
        Mesh::from_raw(
            vertices,
            vec![MeshPrimitive { primitive_type: PrimitiveType::LineList, indices }],
        )
    }

    // ===== intersect_ray_with_lines =====

    #[test]
    fn ray_hits_segment_within_tolerance() {
        // Segment along X at z=5; ray along +Z — perpendicular, distance = 0
        let mesh = make_line_mesh(
            &[(0, 1)],
            vec![make_vertex(-1.0, 0.0, 5.0), make_vertex(1.0, 0.0, 5.0)],
        );
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let hits = intersect_ray_with_lines(&mesh, &ray, 0.1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].segment_index, 0);
        assert!(hits[0].distance_to_ray < 0.01);
    }

    #[test]
    fn ray_misses_segment_outside_tolerance() {
        // Segment 2 units from ray — outside a 0.1 tolerance
        let mesh = make_line_mesh(
            &[(0, 1)],
            vec![make_vertex(-1.0, 2.0, 5.0), make_vertex(1.0, 2.0, 5.0)],
        );
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let hits = intersect_ray_with_lines(&mesh, &ray, 0.1);
        assert!(hits.is_empty());
    }

    #[test]
    fn ray_hits_sorted_nearest_first_by_t() {
        // Two segments at different depths
        let mesh = make_line_mesh(
            &[(0, 1), (2, 3)],
            vec![
                make_vertex(-1.0, 0.0, 10.0), make_vertex(1.0, 0.0, 10.0),
                make_vertex(-1.0, 0.0, 3.0), make_vertex(1.0, 0.0, 3.0),
            ],
        );
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let mut hits = intersect_ray_with_lines(&mesh, &ray, 0.1);
        hits.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
        assert_eq!(hits.len(), 2);
        assert!(hits[0].t < hits[1].t, "closer segment should have smaller t");
        assert_eq!(hits[0].segment_index, 1); // segment at z=3
        assert_eq!(hits[1].segment_index, 0); // segment at z=10
    }

    #[test]
    fn no_line_primitives_returns_empty() {
        let mesh = Mesh::from_raw(
            vec![make_vertex(0.0, 0.0, 0.0), make_vertex(1.0, 0.0, 0.0), make_vertex(0.0, 1.0, 0.0)],
            vec![MeshPrimitive { primitive_type: PrimitiveType::TriangleList, indices: vec![0, 1, 2] }],
        );
        let ray = Ray::new(Point3::new(0.5, 0.5, -1.0), Vector3::new(0.0, 0.0, 1.0));
        let hits = intersect_ray_with_lines(&mesh, &ray, 10.0);
        assert!(hits.is_empty());
    }

    // ===== intersect_volume segment testing =====

    fn unit_box_volume() -> ConvexPolyhedron {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        ConvexPolyhedron::from_aabb(&aabb)
    }

    #[test]
    fn volume_hits_segment_fully_inside() {
        let mesh = make_line_mesh(
            &[(0, 1)],
            vec![make_vertex(-0.5, 0.0, 0.0), make_vertex(0.5, 0.0, 0.0)],
        );
        let result = intersect_volume(&mesh, &unit_box_volume(), false).unwrap();
        assert_eq!(result.segment_indices, vec![0]);
    }

    #[test]
    fn volume_hits_segment_via_midpoint() {
        // Both endpoints are exactly on the boundary (outside), midpoint is inside
        let mesh = make_line_mesh(
            &[(0, 1)],
            vec![make_vertex(-1.5, 0.0, 0.0), make_vertex(1.5, 0.0, 0.0)],
        );
        let result = intersect_volume(&mesh, &unit_box_volume(), false).unwrap();
        assert_eq!(result.segment_indices, vec![0]);
    }

    #[test]
    fn volume_misses_segment_fully_outside() {
        let mesh = make_line_mesh(
            &[(0, 1)],
            vec![make_vertex(2.0, 0.0, 0.0), make_vertex(3.0, 0.0, 0.0)],
        );
        let result = intersect_volume(&mesh, &unit_box_volume(), false);
        assert!(result.is_none());
    }

    #[test]
    fn volume_segment_indices_empty_for_triangle_only_mesh() {
        let mesh = Mesh::from_raw(
            vec![make_vertex(0.0, 0.0, 0.0), make_vertex(0.5, 0.0, 0.0), make_vertex(0.0, 0.5, 0.0)],
            vec![MeshPrimitive { primitive_type: PrimitiveType::TriangleList, indices: vec![0, 1, 2] }],
        );
        let result = intersect_volume(&mesh, &unit_box_volume(), false).unwrap();
        assert!(result.segment_indices.is_empty());
        assert!(!result.triangle_indices.is_empty());
    }
}
