use truck_modeling as truck;

pub type EdgeId = u32;

/// A CAD edge — a parametric curve in 3D space bounded by two vertices.
///
/// Wraps a Truck edge internally. Each edge represents a geometric curve
/// (line, B-spline, NURBS, etc.) connecting two endpoints.
pub struct Edge {
    id: EdgeId,
    inner: truck::Edge,
}

impl Edge {
    pub fn new(id: EdgeId, inner: truck::Edge) -> Self {
        Self { id, inner }
    }

    pub fn id(&self) -> EdgeId {
        self.id
    }

    /// Access the underlying Truck edge.
    pub fn inner(&self) -> &truck::Edge {
        &self.inner
    }
}
