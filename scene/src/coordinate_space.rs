/// The coordinate space in which positions and directions are defined.
///
/// This enum is general-purpose and can be used with lights, annotations,
/// or any scene object that has spatial data.
///
/// # Camera Space
///
/// In camera space, coordinates are relative to the camera:
/// - `(0, 0, 0)` is at the camera eye
/// - `(0, 0, -1)` is the camera forward direction
/// - `(0, 1, 0)` is the camera up direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CoordinateSpace {
    /// World space (default). Positions and directions are absolute.
    #[default]
    World,
    /// Camera space. Positions and directions are relative to the camera.
    Camera,
}
