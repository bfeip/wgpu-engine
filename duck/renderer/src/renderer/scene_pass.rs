mod flat_color;
mod main_pass;
mod outline_pass;
mod silhouette;

pub(crate) use flat_color::{FlatColorPass, FlatColorPassDesc};
pub(crate) use main_pass::{MainPass, OverlayPass};
pub(crate) use outline_pass::OutlinePass;
pub(crate) use silhouette::SilhouetteEdgesPass;
