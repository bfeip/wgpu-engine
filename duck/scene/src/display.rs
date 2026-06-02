//! Per-node render-presentation behavior.
//!
//! [`DisplayBehavior`] groups the placement/presentation modifiers that affect
//! *how* a node's geometry is drawn — distinct from [`crate::NodePayload`] (what
//! a node is) and [`crate::NodeFlags`] (non-visual scene-system semantics like
//! selection, export, bounding).
//!
//! These modifiers inherit down the subtree: setting a layer or screen-space
//! flag on a group root applies to all descendants unless a descendant
//! overrides it. Inheritance is resolved by the renderer during its per-frame
//! traversal, so the scene crate stores only the per-node value and needs no
//! cache. Camera-dependent effects (`screen_sized`, `screen_facing`) are applied
//! at render time and never pollute the node's cached world transform.

/// Which render layer / pass a node's geometry draws in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RenderLayer {
    /// Ordinary scene geometry, depth-tested against the rest of the scene.
    #[default]
    Scene,
    /// Drawn on top of the scene in a separate pass with its own depth buffer,
    /// so it depth-tests among itself but not against scene geometry. Used for
    /// gizmos, handles, and other always-visible annotation geometry.
    Overlay,
    // future: MiniView(MiniViewId) — confined to a sub-viewport of the surface.
}

/// How a node's geometry is presented at render time.
///
/// Defaults to ordinary scene geometry, so a node with the default value is
/// rendered exactly as a node with no special behavior. Inherits down the
/// subtree (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DisplayBehavior {
    /// Keep a constant pixel size regardless of camera distance (the renderer
    /// scales the geometry up as the camera recedes). Applied at render time.
    pub screen_sized: bool,
    /// Orient the geometry to face the camera (billboard): a `look_at` is
    /// applied at render time in addition to the node's own transform.
    pub screen_facing: bool,
    /// Which render layer / pass this node draws in.
    pub layer: RenderLayer,
}
