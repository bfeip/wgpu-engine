//! The built-in [`SnapProvider`]s. Each is a stateless strategy; all live
//! context arrives through [`SnapInput`] and the [`Scene`].

use duck_engine_viewer::common::{transform_point, EuclideanSpace, InnerSpace, Matrix4, Point3};
use duck_engine_viewer::scene::{
    Mesh, NodeFlags, NodePayload, PrimitiveType, Scene, Topology, Visibility,
};

use super::{Snap, SnapInput, SnapKind, SnapFlags, SnapProvider};

/// Free point on the active construction plane. The always-present, lowest-tier
/// fallback so a plane-constrained operator always gets a position.
pub(crate) struct ConstructionPlaneSnap;

impl SnapProvider for ConstructionPlaneSnap {
    fn produces(&self) -> SnapFlags {
        SnapFlags::CONSTRUCTION_PLANE
    }

    fn collect(&self, input: &SnapInput, _scene: &Scene) -> Vec<Snap> {
        if let Some((_, position)) = input.ray.intersect_plane(input.plane) {
            vec![Snap {
                position,
                direction: None,
                kind: SnapKind::ConstructionPlane,
            }]
        } else {
            Vec::new()
        }
    }
}

/// The construction-frame origin. A single high-tier point; ranking decides
/// whether it is near enough to the cursor to win.
pub(crate) struct OriginSnap;

impl SnapProvider for OriginSnap {
    fn produces(&self) -> SnapFlags {
        SnapFlags::ORIGIN
    }

    fn collect(&self, _input: &SnapInput, _scene: &Scene) -> Vec<Snap> {
         return vec![Snap {
            position: Point3::origin(),
            direction: None,
            kind: SnapKind::Origin,
        }];
    }
}

/// The two construction-frame snaps the grid visualizes: the principal axes
/// (U, V, N — higher tier) and gridline-intersection guide points (lower tier).
pub(crate) struct GridSnap;

impl SnapProvider for GridSnap {
    fn produces(&self) -> SnapFlags {
        SnapFlags::GRID_AXIS | SnapFlags::GRID_GUIDE
    }

    fn collect(&self, input: &SnapInput, _scene: &Scene) -> Vec<Snap> {
        let (u, v) = input.plane.basis();
        let origin = input.plane.project_point(Point3::origin());
        let n = input.plane.normal.normalize();
        // Treat the axes as long segments spanning the visible grid.
        let half = (input.grid.size * 0.5).max(input.grid.minor_spacing);

        let mut out = Vec::new();

        // Principal axes through the origin: snap to the closest point on each
        // axis line to the cursor ray (the same primitive edge snapping reuses).
        for dir in [u, v, n] {
            if let Some(approach) = input
                .ray
                .closest_approach_to_segment(origin - dir * half, origin + dir * half)
            {
                out.push(Snap {
                    position: approach.closest_on_segment,
                    direction: Some(dir),
                    kind: SnapKind::GridAxis,
                });
            }
        }

        // Guide point: round the in-plane ray hit to the nearest gridline
        // intersection.
        if let Some((_, hit)) = input.ray.intersect_plane(input.plane) {
            let spacing = input.grid.minor_spacing.max(f32::EPSILON);
            let rel = hit - origin;
            let cu = (rel.dot(u) / spacing).round() * spacing;
            let cv = (rel.dot(v) / spacing).round() * spacing;
            out.push(Snap {
                position: origin + u * cu + v * cv,
                direction: None,
                kind: SnapKind::GridGuide,
            });
        };

        return out;
    }
}

/// B-rep corners (vertices) of existing geometry.
pub(crate) struct CornerSnap;

impl SnapProvider for CornerSnap {
    fn produces(&self) -> SnapFlags {
        SnapFlags::CORNER
    }

    fn collect(&self, input: &SnapInput, scene: &Scene) -> Vec<Snap> {
        // TODO: This is a flat walk over every node. In production this should be
        // a tree walk: it would be more correct (DO_NOT_SELECT / invisibility
        // should propagate to descendants, which a per-node flag check misses)
        // and faster (an excluded or culled parent could prune its whole
        // subtree). It is fine today because parts are added as top-level nodes;
        // revisit when parts gain deep hierarchies.
        let mut snaps = Vec::new();
        for node in scene.nodes() {
            // Skip non-pickable geometry (grid, snap marker), hidden nodes, and
            // anything the caller excluded (e.g. its own in-progress preview).
            if node.flags().contains(NodeFlags::DO_NOT_SELECT) {
                continue;
            }
            if node.visibility() != Visibility::Visible {
                continue;
            }
            if input.exclude_nodes.contains(&node.id) {
                continue;
            }

            let NodePayload::Instance(instance_id) = node.payload() else {
                continue;
            };
            let Some(instance) = scene.get_instance(*instance_id) else {
                continue;
            };
            let Some(mesh) = scene.get_mesh(instance.mesh()) else {
                continue;
            };
            let Some(topology) = mesh.topology() else {
                continue;
            };
            let Some(world) = scene.nodes_transform(node.id) else {
                continue;
            };

            collect_mesh_corners(mesh, topology, &world);
        }

        const EPSILON: f32 = 1e-10;
        snaps.dedup_by(|a: &mut Snap, b: &mut Snap| {
            (a.position - b.position).magnitude() < EPSILON
        });
        return snaps;
    }
}

/// Extracts a mesh's B-rep corner vertices into `out` (in world space).
///
/// Prefers explicit point topology when present; otherwise derives corners from
/// edge-range endpoints — the first/last vertex of each edge is its B-rep vertex,
/// while intermediate curve-approximation points are not corners.
fn collect_mesh_corners(mesh: &Mesh, topology: &Topology, world: &Matrix4) -> Vec<Snap> {
    let mut out = Vec::new();

    let vertices = mesh.vertices();
    let mut push_corner = |local: [f32; 3]| {
        let position = transform_point(world, Point3::new(local[0], local[1], local[2]));
        out.push(Snap {
            position,
            direction: None,
            kind: SnapKind::Corner,
        });
    };

    if !topology.point_ranges.is_empty() {
        // TODO: Only works on first point primitive. Not a big deal right now but...
        if let Some(points) = mesh
            .primitives()
            .iter()
            .find(|p| p.primitive_type == PrimitiveType::PointList)
        {
            for range in &topology.point_ranges {
                for i in range.start..range.start + range.count {
                    if let Some(vtx) = points
                        .indices
                        .get(i as usize)
                        .and_then(|&idx| vertices.get(idx as usize))
                    {
                        push_corner(vtx.position);
                    }
                }
            }
        }
        return out;
    }

    // TODO: Only works on first line primitive. Not a big deal right now but...
    let Some(lines) = mesh
        .primitives()
        .iter()
        .find(|p| p.primitive_type == PrimitiveType::LineList)
    else {
        return out;
    };

    for range in &topology.edge_ranges {
        if range.count == 0 {
            continue;
        }
        // Edge ranges count segments; the LineList holds 2 indices per segment.
        let first = range.start as usize * 2;
        let last = (range.start + range.count - 1) as usize * 2 + 1;
        for slot in [first, last] {
            if let Some(vtx) = lines
                .indices
                .get(slot)
                .and_then(|&idx| vertices.get(idx as usize))
            {
                push_corner(vtx.position);
            }
        }
    }

    return out;
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_viewer::common::{Plane, Ray, Vector3};
    use duck_engine_viewer::scene::{
        Mesh, MeshPrimitive, PositionedCamera, PrimitiveType, SubMeshRange, Topology, Vertex,
    };

    use crate::grid::GridConfig;

    const EPSILON: f32 = 1e-3;

    fn close(a: Point3, b: Point3) -> bool {
        (a - b).magnitude2() < EPSILON * EPSILON
    }

    /// A `SnapInput` whose ray is supplied directly (grid/corner providers don't
    /// use the camera in `collect`, only the ray/plane/grid/scene).
    fn input_with_ray<'a>(
        ray: Ray,
        cam: &'a PositionedCamera,
        plane: &'a Plane,
        grid: &'a GridConfig,
    ) -> SnapInput<'a> {
        SnapInput {
            ray,
            cursor: (0.0, 0.0),
            viewport: (800, 600),
            camera: cam,
            plane,
            grid,
            requested: SnapFlags::all(),
            exclude_nodes: &[],
        }
    }

    fn dummy_camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 10.0, 0.0),
            target: Point3::origin(),
            up: Vector3::new(0.0, 0.0, -1.0),
            aspect: 4.0 / 3.0,
            fovy: 45.0,
            znear: 0.01,
            zfar: 1000.0,
            ortho: false,
        }
    }

    #[test]
    fn grid_guide_rounds_to_nearest_intersection() {
        let cam = dummy_camera();
        let plane = Plane::xz(); // u = +X, v = +Z, origin at world origin
        let grid = GridConfig::default(); // minor_spacing = 5.0
        // Ray straight down hitting the XZ plane at (5.2, 0, -3.1).
        let ray = Ray::new(Point3::new(5.2, 10.0, -3.1), Vector3::new(0.0, -1.0, 0.0));
        let inp = input_with_ray(ray, &cam, &plane, &grid);

        let snaps = GridSnap.collect(&inp, &Scene::new());

        let guide = snaps
            .iter()
            .find(|s| s.kind == SnapKind::GridGuide)
            .expect("expected a grid guide candidate");
        // 5.2 → 5.0, −3.1 → −5.0
        assert!(close(guide.position, Point3::new(5.0, 0.0, -5.0)), "{:?}", guide.position);
    }

    #[test]
    fn grid_axis_snaps_to_principal_axis_line() {
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        // Vertical ray near the X axis: the X-axis closest point is directly
        // below the ray's x, on the axis (y = z = 0).
        let ray = Ray::new(Point3::new(7.0, 10.0, -1.5), Vector3::new(0.0, -1.0, 0.0));
        let inp = input_with_ray(ray, &cam, &plane, &grid);

        let snaps = GridSnap.collect(&inp, &Scene::new());

        // Three principal axes (U, V, N) should each yield a candidate.
        let axes: Vec<_> = snaps.iter().filter(|s| s.kind == SnapKind::GridAxis).collect();
        assert_eq!(axes.len(), 3);

        // The U (X) axis candidate sits at (7, 0, 0) with direction ±X.
        let x_axis = axes
            .iter()
            .find(|s| s.direction.map_or(false, |d| d.x.abs() > 0.9))
            .expect("expected an X-axis candidate");
        assert!(close(x_axis.position, Point3::new(7.0, 0.0, 0.0)), "{:?}", x_axis.position);
    }

    #[test]
    fn corners_from_edge_endpoints_in_world_space() {
        // One straight edge A→B as a single LineList segment.
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mesh = Mesh::from_raw(
            vec![
                Vertex { position: a, tex_coords: [0.0; 3], normal: [0.0; 3] },
                Vertex { position: b, tex_coords: [0.0; 3], normal: [0.0; 3] },
            ],
            vec![MeshPrimitive {
                primitive_type: PrimitiveType::LineList,
                indices: vec![0, 1],
            }],
        );
        let topology = Topology {
            face_ranges: vec![],
            edge_ranges: vec![SubMeshRange { start: 0, count: 1 }],
            point_ranges: vec![],
        };
        // Translate the whole part by +10 in X.
        let world = Matrix4::from_translation(Vector3::new(10.0, 0.0, 0.0));

        let snaps = collect_mesh_corners(&mesh, &topology, &world);

        assert_eq!(snaps.len(), 2);
        assert!(snaps.iter().any(|s| close(s.position, Point3::new(11.0, 2.0, 3.0))));
        assert!(snaps.iter().any(|s| close(s.position, Point3::new(14.0, 5.0, 6.0))));
        assert!(snaps.iter().all(|s| s.kind == SnapKind::Corner));
    }
}
