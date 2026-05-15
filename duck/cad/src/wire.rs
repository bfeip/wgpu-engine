use glam::DVec3;
use opencascade::primitives::{Edge, Wire};

/// An opaque handle to a B-Rep edge (curve segment). Used to build [`CadWire`]s.
pub struct CadEdge {
    pub(crate) inner: Edge,
}

impl CadEdge {
    /// A straight line segment between two points.
    pub fn line(from: [f64; 3], to: [f64; 3]) -> Self {
        Self { inner: Edge::segment(DVec3::from(from), DVec3::from(to)) }
    }

    /// A circular arc passing through three points.
    ///
    /// `p1` and `p3` are the endpoints; `mid` is a point on the arc between them.
    pub fn arc(p1: [f64; 3], mid: [f64; 3], p2: [f64; 3]) -> Self {
        Self {
            inner: Edge::arc(DVec3::from(p1), DVec3::from(mid), DVec3::from(p2)),
        }
    }

    /// A full circle with the given centre, normal, and radius.
    pub fn circle(center: [f64; 3], normal: [f64; 3], radius: f64) -> Self {
        Self {
            inner: Edge::circle(DVec3::from(center), DVec3::from(normal), radius),
        }
    }

    /// A Bezier curve defined by control points.
    pub fn bezier(control_points: &[[f64; 3]]) -> Self {
        let pts = control_points.iter().copied().map(DVec3::from);
        Self { inner: Edge::bezier(pts) }
    }

    /// A B-spline curve interpolated through the given points.
    ///
    /// Optionally provide `(start_tangent, end_tangent)` to constrain the curve ends.
    pub fn spline(points: &[[f64; 3]], tangents: Option<([f64; 3], [f64; 3])>) -> Self {
        let pts = points.iter().copied().map(DVec3::from);
        let tangents = tangents.map(|(t0, t1)| (DVec3::from(t0), DVec3::from(t1)));
        Self { inner: Edge::spline_from_points(pts, tangents) }
    }
}

/// An opaque handle to a B-Rep wire (an ordered sequence of connected edges).
/// Used as a profile for extrusion or revolution via [`CadShape::extrude`] /
/// [`CadShape::revolve`].
pub struct CadWire {
    pub(crate) inner: Wire,
}

impl CadWire {
    /// Build a wire from an ordered sequence of edges.
    ///
    /// Edges must be connected end-to-end within OCCT's default fuzzy tolerance.
    pub fn from_edges(edges: &[CadEdge]) -> Self {
        let occ_edges: Vec<&Edge> = edges.iter().map(|e| &e.inner).collect();
        Self { inner: Wire::from_edges(occ_edges) }
    }

    /// A closed rectangular wire centred at the origin in the XY plane.
    pub fn rect(width: f64, height: f64) -> Self {
        Self { inner: Wire::rect(width, height) }
    }
}
