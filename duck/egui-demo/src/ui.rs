mod environment_tab;
mod info_sections;
mod left_panel;
mod lights_tab;
mod right_panel;
mod scene_tab;

pub use left_panel::LeftPanel;
pub use right_panel::RightPanel;

use duck_engine_viewer::operator::NavigationMode;
use duck_engine_viewer::scene::{Camera, LightType, NodeId, Visibility};
use duck_engine_viewer::Viewer;

/// A visibility change requested by the UI.
pub struct VisibilityChange {
    pub node_id: NodeId,
    pub new_visibility: Visibility,
}

/// Actions requested by the UI that need to be handled by the application.
#[derive(Default)]
pub struct UiActions {
    pub load_scene: bool,
    pub save_scene: bool,
    pub clear_scene: bool,
    pub add_light: Option<LightType>,
    pub load_environment: bool,
    pub clear_environment: bool,
    pub visibility_changes: Vec<VisibilityChange>,
    /// When set, replace the active camera with this value.
    pub set_camera: Option<Camera>,
}

/// Information about the current navigation mode, shared across panels.
pub struct ModeInfo {
    pub mode: NavigationMode,
}

/// Persistent UI state stored in the application.
#[derive(Default)]
pub struct UiState {
    pub left: LeftPanel,
    pub right: RightPanel,
}

impl UiState {
    pub fn build(&mut self, ctx: &egui::Context, viewer: &mut Viewer) -> UiActions {
        let mut actions = UiActions::default();
        let mode_info = ModeInfo { mode: viewer.navigation_mode() };

        build_performance_panel(ctx, &mode_info);
        self.left.show(ctx, viewer, &mut actions);
        self.right.show(ctx, viewer, &mode_info);

        actions
    }
}

fn build_performance_panel(ctx: &egui::Context, mode: &ModeInfo) {
    egui::TopBottomPanel::new(egui::panel::TopBottomSide::Top, "Performance")
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("FPS: {:.1}", ctx.input(|i| i.stable_dt).recip()));
                ui.separator();
                match mode.mode {
                    NavigationMode::Walk => ui.label("Mode: Walk"),
                    NavigationMode::Orbit => ui.label("Mode: Orbit"),
                };
            });
        });
}
