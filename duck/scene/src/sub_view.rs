//! Sub-views: rectangular sub-regions of the surface that render a separate
//! subtree of the scene through a separate camera.
//!
//! A [`SubView`] names a [`ViewportRect`] (its placement on the surface, as
//! proportions in `[0, 1]`), the [`NodeId`] whose subtree it draws, and the
//! camera [`NodeId`] it views that subtree through. The subtree is drawn *only*
//! inside the sub-view region — the renderer excludes it from the main pass.
//!
//! The scene stores only this device-free definition; the renderer resolves the
//! pixel rectangle and the [`crate::PositionedCamera`] each frame from the
//! current surface size.

use crate::NodeId;

/// Unique identifier for a [`SubView`]. Backed by UUID v7 like [`NodeId`].
pub type SubViewId = crate::Id<SubView>;

/// A sub-view's placement on the surface, as proportions of the full surface in
/// the range `[0, 1]`. `(x, y)` is the lower-left-relative origin convention used
/// by the renderer's viewport/scissor (top-left in window space is handled by the
/// renderer when converting to pixels).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ViewportRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl ViewportRect {
    /// A rectangle covering the entire surface.
    pub const FULL: ViewportRect = ViewportRect { x: 0.0, y: 0.0, width: 1.0, height: 1.0 };

    /// Creates a new viewport rectangle from proportions in `[0, 1]`.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    /// Resolves this proportional rectangle to integer pixel coordinates against a
    /// surface of `(surface_width, surface_height)` pixels. Returns
    /// `(x, y, width, height)` in pixels, clamped so the rectangle stays within the
    /// surface and has a non-zero extent.
    pub fn to_pixels(&self, surface_width: u32, surface_height: u32) -> (u32, u32, u32, u32) {
        let sw = surface_width as f32;
        let sh = surface_height as f32;
        let x = (self.x * sw).round().clamp(0.0, sw);
        let y = (self.y * sh).round().clamp(0.0, sh);
        let w = (self.width * sw).round().clamp(1.0, (sw - x).max(1.0));
        let h = (self.height * sh).round().clamp(1.0, (sh - y).max(1.0));
        (x as u32, y as u32, w as u32, h as u32)
    }
}

/// A sub-region of the surface that renders one subtree through one camera node.
///
/// See the [module docs](self) for the rendering semantics.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SubView {
    pub id: SubViewId,
    /// Placement on the surface, as a proportion of the full surface.
    pub rect: ViewportRect,
    /// Root of the subtree drawn in this sub-view (excluded from the main pass).
    pub root: NodeId,
    /// A node carrying a [`crate::NodePayload::Camera`] used to view the subtree.
    pub camera: NodeId,
}

impl SubView {
    /// Creates a new sub-view with a freshly generated id.
    pub fn new(rect: ViewportRect, root: NodeId, camera: NodeId) -> Self {
        Self { id: SubViewId::new(), rect, root, camera }
    }
}

#[cfg(test)]
mod tests {
    use crate::{CameraProjection, NodeFlags, NodePayload, Scene, ViewportRect, common};

    const EPSILON: f32 = 1e-4;

    #[test]
    fn add_and_query_sub_view() {
        let mut scene = Scene::new();
        let root = scene
            .add_node(None, Some("sv_root".into()), common::Transform::IDENTITY, NodeFlags::NONE)
            .unwrap();
        let camera = scene
            .add_node(None, Some("sv_cam".into()), common::Transform::IDENTITY, NodeFlags::NONE)
            .unwrap();

        let rect = ViewportRect::new(0.7, 0.0, 0.3, 0.3);
        let id = scene.add_sub_view(rect, root, camera);

        assert_eq!(scene.sub_view_count(), 1);
        let sv = scene.get_sub_view(id).expect("sub-view present");
        assert_eq!(sv.rect, rect);
        assert_eq!(sv.root, root);
        assert_eq!(sv.camera, camera);

        let new_rect = ViewportRect::new(0.0, 0.0, 0.5, 0.5);
        scene.set_sub_view_rect(id, new_rect);
        assert_eq!(scene.get_sub_view(id).unwrap().rect, new_rect);

        scene.remove_sub_view(id);
        assert_eq!(scene.sub_view_count(), 0);
    }

    #[test]
    fn positioned_camera_uses_given_aspect() {
        let mut scene = Scene::new();
        let camera = scene
            .add_node(None, Some("cam".into()), common::Transform::IDENTITY, NodeFlags::NONE)
            .unwrap();
        let proj = CameraProjection {
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: false,
            focus_distance: 5.0,
        };
        scene.set_node_payload(camera, NodePayload::Camera(proj));

        let positioned = scene.positioned_camera_for_node(camera, 2.0).expect("camera resolves");
        assert!((positioned.aspect - 2.0).abs() < EPSILON);
    }

    #[test]
    fn viewport_rect_to_pixels_clamps_within_surface() {
        let rect = ViewportRect::new(0.5, 0.5, 0.6, 0.6);
        let (x, y, w, h) = rect.to_pixels(100, 100);
        assert_eq!((x, y), (50, 50));
        // width/height clamped so the rect stays inside the surface.
        assert_eq!((w, h), (50, 50));
    }
}
