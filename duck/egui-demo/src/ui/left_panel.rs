use duck_engine_viewer::Viewer;

use super::{environment_tab, lights_tab, scene_tab, UiActions};
#[cfg(feature = "streaming")]
use super::network_tab::{self, NetworkTabState};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LeftPanelTab {
    #[default]
    Scene,
    Lights,
    Environment,
    #[cfg(feature = "streaming")]
    Network,
}

#[derive(Default)]
pub struct LeftPanel {
    pub active_tab: LeftPanelTab,
    #[cfg(feature = "streaming")]
    pub network: NetworkTabState,
}

impl LeftPanel {
    pub fn show(&mut self, ctx: &egui::Context, viewer: &mut Viewer, actions: &mut UiActions) {
        egui::SidePanel::new(egui::panel::Side::Left, "Left Panel")
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.active_tab, LeftPanelTab::Scene, "Scene");
                    ui.selectable_value(&mut self.active_tab, LeftPanelTab::Lights, "Lights");
                    ui.selectable_value(&mut self.active_tab, LeftPanelTab::Environment, "Env");
                    #[cfg(feature = "streaming")]
                    ui.selectable_value(&mut self.active_tab, LeftPanelTab::Network, "Network");
                });

                ui.separator();

                match self.active_tab {
                    LeftPanelTab::Scene => scene_tab::show(ui, viewer, actions),
                    LeftPanelTab::Lights => lights_tab::show(ui, viewer, actions),
                    LeftPanelTab::Environment => environment_tab::show(ui, viewer, actions),
                    #[cfg(feature = "streaming")]
                    LeftPanelTab::Network => network_tab::show(ui, viewer, actions, &mut self.network),
                }
            });
    }
}
