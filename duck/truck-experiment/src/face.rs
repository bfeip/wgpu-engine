use truck_modeling as truck;

pub type FaceId = u32;

/// A CAD face — a trimmed surface in 3D space bounded by one or more wires (edge loops).
///
/// Wraps a Truck face internally. Each face represents a geometric surface
/// (plane, B-spline, NURBS, etc.) trimmed by boundary curves.
pub struct Face {
    id: FaceId,
    inner: truck::Face,
}

impl Face {
    pub fn new(id: FaceId, inner: truck::Face) -> Self {
        Self { id, inner }
    }

    pub fn id(&self) -> FaceId {
        self.id
    }

    /// Access the underlying Truck face.
    pub fn inner(&self) -> &truck::Face {
        &self.inner
    }
}
