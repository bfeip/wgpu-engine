//! Centralized icon assets for the modeler's panels. Each icon is a `(uri, bytes)`
//! pair: the URI is a stable cache key for egui, the bytes are the embedded SVG.
//! Icons are authored white-filled so they can be tinted per use site.
//!

// TODO: Tool palette icons still live on each tool's `ToolInfo`; folding them in here is
// a worthwhile future cleanup.

use crate::document::PartKind;

/// An embedded SVG: `(cache uri, bytes)`.
pub type Icon = (&'static str, &'static [u8]);

/// Embed `assets/svg/<file>`, deriving the egui cache URI from the file name.
macro_rules! icon {
    ($file:expr) => {
        (
            concat!("bytes://", $file),
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/svg/", $file)),
        )
    };
}

pub const EYE: Icon = icon!("eye.svg");
pub const EYE_OFF: Icon = icon!("eye-off.svg");

pub const KIND_SOLID: Icon = icon!("kind-solid.svg");
pub const KIND_FACE: Icon = icon!("kind-face.svg");
pub const KIND_WIRE: Icon = icon!("kind-wire.svg");
pub const KIND_POINT: Icon = icon!("kind-point.svg");
pub const KIND_OTHER: Icon = icon!("cube.svg");

/// Leading icon for a part row, by kind.
pub fn kind_icon(kind: PartKind) -> Icon {
    match kind {
        PartKind::Solid => KIND_SOLID,
        // Shells and faces share the surface glyph.
        PartKind::Shell | PartKind::Face => KIND_FACE,
        PartKind::Wire => KIND_WIRE,
        PartKind::Point => KIND_POINT,
        PartKind::Compound | PartKind::Other => KIND_OTHER,
    }
}

/// Accent color for a part's kind icon and badge.
pub fn kind_color(kind: PartKind) -> egui::Color32 {
    match kind {
        PartKind::Solid => egui::Color32::from_rgb(0x6E, 0x9B, 0xF0),
        PartKind::Shell | PartKind::Face => egui::Color32::from_rgb(0xE0, 0xA1, 0x4A),
        PartKind::Wire => egui::Color32::from_rgb(0x5C, 0xC2, 0x9A),
        PartKind::Point => egui::Color32::from_rgb(0xC8, 0x7C, 0xD8),
        PartKind::Compound | PartKind::Other => egui::Color32::from_rgb(0x9A, 0x9A, 0x9A),
    }
}
