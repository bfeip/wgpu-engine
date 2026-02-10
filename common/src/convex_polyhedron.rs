use cgmath::{InnerSpace, Matrix4, Point3, Vector3};

use crate::{Aabb, Plane, Ray, EPSILON};

/// A convex polyhedron defined as the intersection of half-spaces (planes).
///
/// A point is inside the polyhedron if it is on the "inside" (negative signed distance)
/// of all planes. This representation is efficient for frustum culling and volume picking.
#[derive(Debug, Clone)]
pub struct ConvexPolyhedron {
    planes: Vec<Plane>,
}

impl ConvexPolyhedron {
    /// Creates a new convex polyhedron from a list of planes.
    /// Each plane's normal should point "outward" from the volume.
    pub fn new(planes: Vec<Plane>) -> Self {
        Self { planes }
    }

    /// Returns the planes defining this polyhedron.
    pub fn planes(&self) -> &[Plane] {
        &self.planes
    }

    /// Extracts 6 frustum planes from a view-projection matrix.
    ///
    /// Uses the Gribb/Hartmann method to extract planes from a combined
    /// view-projection matrix. The planes are ordered: Left, Right, Bottom, Top, Near, Far.
    ///
    /// The normals point inward (toward the center of the frustum), so we negate them
    /// to follow our convention of normals pointing outward.
    pub fn from_frustum(view_proj: &Matrix4<f32>) -> Self {
        // Extract rows from the matrix (cgmath is column-major, so we access columns)
        let row0 = Vector3::new(view_proj[0][0], view_proj[1][0], view_proj[2][0]);
        let row1 = Vector3::new(view_proj[0][1], view_proj[1][1], view_proj[2][1]);
        let row2 = Vector3::new(view_proj[0][2], view_proj[1][2], view_proj[2][2]);
        let row3 = Vector3::new(view_proj[0][3], view_proj[1][3], view_proj[2][3]);

        let w0 = view_proj[3][0];
        let w1 = view_proj[3][1];
        let w2 = view_proj[3][2];
        let w3 = view_proj[3][3];

        // Gribb/Hartmann frustum plane extraction
        // The planes point inward by default, but our convention uses outward normals,
        // so we negate them.
        let planes = vec![
            // Left: row3 + row0
            Plane::from_coefficients(
                -(row3.x + row0.x),
                -(row3.y + row0.y),
                -(row3.z + row0.z),
                -(w3 + w0),
            ),
            // Right: row3 - row0
            Plane::from_coefficients(
                -(row3.x - row0.x),
                -(row3.y - row0.y),
                -(row3.z - row0.z),
                -(w3 - w0),
            ),
            // Bottom: row3 + row1
            Plane::from_coefficients(
                -(row3.x + row1.x),
                -(row3.y + row1.y),
                -(row3.z + row1.z),
                -(w3 + w1),
            ),
            // Top: row3 - row1
            Plane::from_coefficients(
                -(row3.x - row1.x),
                -(row3.y - row1.y),
                -(row3.z - row1.z),
                -(w3 - w1),
            ),
            // Near: row3 + row2
            Plane::from_coefficients(
                -(row3.x + row2.x),
                -(row3.y + row2.y),
                -(row3.z + row2.z),
                -(w3 + w2),
            ),
            // Far: row3 - row2
            Plane::from_coefficients(
                -(row3.x - row2.x),
                -(row3.y - row2.y),
                -(row3.z - row2.z),
                -(w3 - w2),
            ),
        ];

        Self { planes }
    }

    /// Creates a convex polyhedron from an axis-aligned bounding box.
    /// Results in 6 planes (one for each face of the box).
    pub fn from_aabb(aabb: &Aabb) -> Self {
        let planes = vec![
            // -X face (normal points -X, outward from box)
            Plane::new(Vector3::new(-1.0, 0.0, 0.0), Point3::new(aabb.min.x, 0.0, 0.0)),
            // +X face
            Plane::new(Vector3::new(1.0, 0.0, 0.0), Point3::new(aabb.max.x, 0.0, 0.0)),
            // -Y face
            Plane::new(Vector3::new(0.0, -1.0, 0.0), Point3::new(0.0, aabb.min.y, 0.0)),
            // +Y face
            Plane::new(Vector3::new(0.0, 1.0, 0.0), Point3::new(0.0, aabb.max.y, 0.0)),
            // -Z face
            Plane::new(Vector3::new(0.0, 0.0, -1.0), Point3::new(0.0, 0.0, aabb.min.z)),
            // +Z face
            Plane::new(Vector3::new(0.0, 0.0, 1.0), Point3::new(0.0, 0.0, aabb.max.z)),
        ];

        Self { planes }
    }

    /// Tests if a point is inside the polyhedron (on the inside of all planes).
    pub fn contains_point(&self, point: Point3<f32>) -> bool {
        self.planes
            .iter()
            .all(|plane| plane.signed_distance(point) <= EPSILON)
    }

    /// Tests if an AABB intersects with the polyhedron (broad-phase test).
    ///
    /// This is a conservative test: it may return true for AABBs that don't actually
    /// intersect (false positives), but it will never return false for AABBs that do
    /// intersect (no false negatives). This makes it suitable for broad-phase culling.
    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        let corners = aabb.corners();

        for plane in &self.planes {
            // If all 8 corners are outside this plane, the AABB is completely outside
            let all_outside = corners
                .iter()
                .all(|corner| plane.signed_distance(*corner) > EPSILON);

            if all_outside {
                return false;
            }
        }

        // Conservative: AABB might intersect
        true
    }

    /// Tests if an AABB is fully contained within the polyhedron.
    pub fn contains_aabb(&self, aabb: &Aabb) -> bool {
        let corners = aabb.corners();
        corners.iter().all(|corner| self.contains_point(*corner))
    }

    /// Tests if a triangle intersects the polyhedron.
    ///
    /// # Arguments
    /// * `v0`, `v1`, `v2` - The triangle vertices
    /// * `thorough` - If true, also tests polyhedron edges against the triangle.
    ///   This catches edge cases where the polyhedron passes through the triangle
    ///   without any triangle vertex being inside and without triangle edges crossing
    ///   the polyhedron boundary. This is more expensive but more accurate.
    pub fn intersects_triangle(
        &self,
        v0: Point3<f32>,
        v1: Point3<f32>,
        v2: Point3<f32>,
        thorough: bool,
    ) -> bool {
        // Test 1: Any vertex inside the polyhedron?
        if self.contains_point(v0) || self.contains_point(v1) || self.contains_point(v2) {
            return true;
        }

        // Test 2: Any triangle edge crosses into the polyhedron?
        let edges = [(v0, v1), (v1, v2), (v2, v0)];
        for (start, end) in edges {
            if self.edge_enters_volume(start, end) {
                return true;
            }
        }

        // Test 3 (thorough only): Polyhedron edges cross the triangle?
        if thorough {
            if self.polyhedron_edges_intersect_triangle(v0, v1, v2) {
                return true;
            }
        }

        false
    }

    /// Tests if a triangle is fully contained within the polyhedron.
    pub fn contains_triangle(
        &self,
        v0: Point3<f32>,
        v1: Point3<f32>,
        v2: Point3<f32>,
    ) -> bool {
        self.contains_point(v0) && self.contains_point(v1) && self.contains_point(v2)
    }

    /// Tests if an edge enters the volume (crosses from outside to inside).
    fn edge_enters_volume(&self, start: Point3<f32>, end: Point3<f32>) -> bool {
        // For each plane, check if the edge crosses it and if so,
        // check if the crossing point is inside all other planes
        for (i, plane) in self.planes.iter().enumerate() {
            if let Some((_, intersection_point)) = plane.intersect_segment(start, end) {
                // Check if intersection point is inside all OTHER planes
                let inside_others = self.planes.iter().enumerate().all(|(j, other_plane)| {
                    i == j || other_plane.signed_distance(intersection_point) <= EPSILON
                });

                if inside_others {
                    return true;
                }
            }
        }

        false
    }

    /// Tests if any polyhedron edge intersects the triangle.
    /// This requires computing the polyhedron's edges from plane intersections.
    fn polyhedron_edges_intersect_triangle(
        &self,
        v0: Point3<f32>,
        v1: Point3<f32>,
        v2: Point3<f32>,
    ) -> bool {
        // For a frustum or simple convex polyhedron, we can compute edges
        // by finding intersections of plane pairs that form edges.
        //
        // For a frustum specifically, we have 12 edges:
        // - 4 edges on near plane
        // - 4 edges on far plane
        // - 4 edges connecting near to far
        //
        // For general convex polyhedra, this is more complex.
        // We use a simpler approach: sample rays from inside the polyhedron
        // toward the triangle and see if any hit.

        // Compute triangle normal and plane
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let tri_normal = edge1.cross(edge2);

        if tri_normal.magnitude2() < EPSILON * EPSILON {
            return false; // Degenerate triangle
        }

        // For each pair of adjacent planes, find their intersection line
        // and test if it passes through the triangle
        let n = self.planes.len();
        for i in 0..n {
            for j in (i + 1)..n {
                // Find the line of intersection between planes i and j
                if let Some((line_point, line_dir)) =
                    plane_plane_intersection(&self.planes[i], &self.planes[j])
                {
                    // Find where this line intersects the triangle's plane
                    let ray = Ray::new(line_point, line_dir);
                    if let Some((t, _, _)) = ray.intersect_triangle(v0, v1, v2) {
                        // Check if this intersection point is inside the polyhedron
                        let hit_point = ray.point_at(t);
                        if self.contains_point(hit_point) {
                            return true;
                        }
                    }

                    // Also test the opposite direction
                    let ray_neg = Ray::new(line_point, -line_dir);
                    if let Some((t, _, _)) = ray_neg.intersect_triangle(v0, v1, v2) {
                        let hit_point = ray_neg.point_at(t);
                        if self.contains_point(hit_point) {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    /// Transforms all planes by the given matrix.
    pub fn transform(&self, matrix: &Matrix4<f32>) -> Self {
        let transformed_planes = self
            .planes
            .iter()
            .map(|plane| plane.transform(matrix))
            .collect();

        Self {
            planes: transformed_planes,
        }
    }
}

/// Finds the intersection line of two planes.
/// Returns Some((point_on_line, direction)) or None if planes are parallel.
fn plane_plane_intersection(p1: &Plane, p2: &Plane) -> Option<(Point3<f32>, Vector3<f32>)> {
    use cgmath::InnerSpace;

    // Direction of intersection line is perpendicular to both normals
    let direction = p1.normal.cross(p2.normal);

    // If normals are parallel, planes don't intersect in a line
    if direction.magnitude2() < EPSILON * EPSILON {
        return None;
    }

    let direction = direction.normalize();

    // Find a point on the line by solving the system:
    // n1 · p = -d1
    // n2 · p = -d2
    // We can find a point by setting one coordinate to 0 and solving for the other two

    // Use the axis most perpendicular to the line direction
    let abs_dir = Vector3::new(direction.x.abs(), direction.y.abs(), direction.z.abs());

    let point = if abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z {
        // Set x = 0, solve for y and z
        let denom = p1.normal.y * p2.normal.z - p1.normal.z * p2.normal.y;
        if denom.abs() < EPSILON {
            return None;
        }
        let y = (-p1.distance * p2.normal.z + p2.distance * p1.normal.z) / denom;
        let z = (-p1.normal.y * p2.distance + p2.normal.y * p1.distance) / denom;
        Point3::new(0.0, y, z)
    } else if abs_dir.y >= abs_dir.z {
        // Set y = 0, solve for x and z
        let denom = p1.normal.x * p2.normal.z - p1.normal.z * p2.normal.x;
        if denom.abs() < EPSILON {
            return None;
        }
        let x = (-p1.distance * p2.normal.z + p2.distance * p1.normal.z) / denom;
        let z = (-p1.normal.x * p2.distance + p2.normal.x * p1.distance) / denom;
        Point3::new(x, 0.0, z)
    } else {
        // Set z = 0, solve for x and y
        let denom = p1.normal.x * p2.normal.y - p1.normal.y * p2.normal.x;
        if denom.abs() < EPSILON {
            return None;
        }
        let x = (-p1.distance * p2.normal.y + p2.distance * p1.normal.y) / denom;
        let y = (-p1.normal.x * p2.distance + p2.normal.x * p1.distance) / denom;
        Point3::new(x, y, 0.0)
    };

    Some((point, direction))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Deg, EuclideanSpace, Matrix4, PerspectiveFov, Point3, Rad, SquareMatrix, Vector3};

    fn create_unit_cube_polyhedron() -> ConvexPolyhedron {
        // Create a unit cube centered at origin (-0.5 to 0.5 on each axis)
        let aabb = Aabb::new(
            Point3::new(-0.5, -0.5, -0.5),
            Point3::new(0.5, 0.5, 0.5),
        );
        ConvexPolyhedron::from_aabb(&aabb)
    }

    #[test]
    fn test_polyhedron_from_aabb() {
        let poly = create_unit_cube_polyhedron();
        assert_eq!(poly.planes.len(), 6);
    }

    #[test]
    fn test_contains_point_inside() {
        let poly = create_unit_cube_polyhedron();

        // Center should be inside
        assert!(poly.contains_point(Point3::origin()));

        // Points inside should be contained
        assert!(poly.contains_point(Point3::new(0.25, 0.25, 0.25)));
        assert!(poly.contains_point(Point3::new(-0.25, -0.25, -0.25)));
    }

    #[test]
    fn test_contains_point_outside() {
        let poly = create_unit_cube_polyhedron();

        // Points outside should not be contained
        assert!(!poly.contains_point(Point3::new(1.0, 0.0, 0.0)));
        assert!(!poly.contains_point(Point3::new(0.0, 1.0, 0.0)));
        assert!(!poly.contains_point(Point3::new(0.0, 0.0, 1.0)));
        assert!(!poly.contains_point(Point3::new(0.0, 0.0, -1.0)));
    }

    #[test]
    fn test_contains_point_on_surface() {
        let poly = create_unit_cube_polyhedron();

        // Points on surface should be considered inside (within epsilon)
        assert!(poly.contains_point(Point3::new(0.5, 0.0, 0.0)));
        assert!(poly.contains_point(Point3::new(0.0, 0.5, 0.0)));
        assert!(poly.contains_point(Point3::new(0.0, 0.0, 0.5)));
    }

    #[test]
    fn test_intersects_aabb_inside() {
        let poly = create_unit_cube_polyhedron();

        // Small AABB fully inside
        let small_aabb = Aabb::new(
            Point3::new(-0.1, -0.1, -0.1),
            Point3::new(0.1, 0.1, 0.1),
        );
        assert!(poly.intersects_aabb(&small_aabb));
    }

    #[test]
    fn test_intersects_aabb_overlapping() {
        let poly = create_unit_cube_polyhedron();

        // AABB that partially overlaps
        let overlapping_aabb = Aabb::new(
            Point3::new(0.25, 0.25, 0.25),
            Point3::new(0.75, 0.75, 0.75),
        );
        assert!(poly.intersects_aabb(&overlapping_aabb));
    }

    #[test]
    fn test_intersects_aabb_outside() {
        let poly = create_unit_cube_polyhedron();

        // AABB completely outside
        let outside_aabb = Aabb::new(
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(2.0, 2.0, 2.0),
        );
        assert!(!poly.intersects_aabb(&outside_aabb));
    }

    #[test]
    fn test_contains_triangle_inside() {
        let poly = create_unit_cube_polyhedron();

        // Small triangle inside
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(0.1, 0.0, 0.0);
        let v2 = Point3::new(0.0, 0.1, 0.0);

        assert!(poly.contains_triangle(v0, v1, v2));
    }

    #[test]
    fn test_contains_triangle_outside() {
        let poly = create_unit_cube_polyhedron();

        // Triangle outside
        let v0 = Point3::new(2.0, 0.0, 0.0);
        let v1 = Point3::new(2.1, 0.0, 0.0);
        let v2 = Point3::new(2.0, 0.1, 0.0);

        assert!(!poly.contains_triangle(v0, v1, v2));
    }

    #[test]
    fn test_intersects_triangle_vertex_inside() {
        let poly = create_unit_cube_polyhedron();

        // Triangle with one vertex inside
        let v0 = Point3::new(0.0, 0.0, 0.0); // Inside
        let v1 = Point3::new(2.0, 0.0, 0.0); // Outside
        let v2 = Point3::new(0.0, 2.0, 0.0); // Outside

        assert!(poly.intersects_triangle(v0, v1, v2, false));
    }

    #[test]
    fn test_intersects_triangle_edge_crosses() {
        let poly = create_unit_cube_polyhedron();

        // Triangle with edge crossing through the cube
        let v0 = Point3::new(-2.0, 0.0, 0.0); // Outside left
        let v1 = Point3::new(2.0, 0.0, 0.0);  // Outside right
        let v2 = Point3::new(0.0, 2.0, 0.0);  // Outside top

        // This triangle has an edge that passes through the cube
        assert!(poly.intersects_triangle(v0, v1, v2, false));
    }

    #[test]
    fn test_intersects_triangle_no_intersection() {
        let poly = create_unit_cube_polyhedron();

        // Triangle completely outside
        let v0 = Point3::new(2.0, 0.0, 0.0);
        let v1 = Point3::new(3.0, 0.0, 0.0);
        let v2 = Point3::new(2.5, 1.0, 0.0);

        assert!(!poly.intersects_triangle(v0, v1, v2, false));
    }

    #[test]
    fn test_transform_translation() {
        let poly = create_unit_cube_polyhedron();
        let translation = Matrix4::from_translation(Vector3::new(5.0, 0.0, 0.0));

        let transformed = poly.transform(&translation);

        // Origin should no longer be inside
        assert!(!transformed.contains_point(Point3::origin()));

        // (5, 0, 0) should now be inside
        assert!(transformed.contains_point(Point3::new(5.0, 0.0, 0.0)));
    }

    #[test]
    fn test_from_frustum_basic() {
        // Create a simple perspective projection
        let fov: PerspectiveFov<f32> = PerspectiveFov {
            fovy: Rad::from(Deg(90.0)),
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
        };
        let proj: Matrix4<f32> = fov.into();

        // Simple view matrix (identity = camera at origin looking down -Z)
        let view = Matrix4::identity();
        let view_proj = proj * view;

        let frustum = ConvexPolyhedron::from_frustum(&view_proj);

        assert_eq!(frustum.planes.len(), 6);

        // Point inside frustum (in front of camera, within FOV)
        // With 90 degree FOV and camera at origin looking down -Z,
        // a point at (0, 0, -1) should be inside
        assert!(frustum.contains_point(Point3::new(0.0, 0.0, -1.0)));

        // Point behind camera should be outside
        assert!(!frustum.contains_point(Point3::new(0.0, 0.0, 1.0)));
    }
}
