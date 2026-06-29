//! The right-hand tabbed panel (Scene / Props / Material / Light / Snap).
//! Only the Scene tab is populated today; the others render a placeholder.

use duck_engine_viewer::selection::SelectionManager;

use crate::document::Document;
use crate::ui::scene_tab::SceneTab;

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum RightTab {
    #[default]
    Scene,
    Props,
    Material,
    Light,
    Snap,
}

impl RightTab {
    const ALL: [RightTab; 5] = [
        RightTab::Scene,
        RightTab::Props,
        RightTab::Material,
        RightTab::Light,
        RightTab::Snap,
    ];

    fn label(self) -> &'static str {
        match self {
            RightTab::Scene => "Scene",
            RightTab::Props => "Props",
            RightTab::Material => "Material",
            RightTab::Light => "Light",
            RightTab::Snap => "Snap",
        }
    }
}

#[derive(Default)]
pub struct RightPanel {
    active_tab: RightTab,
    scene: SceneTab,
}

impl RightPanel {
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        document: &mut Document,
        selection: &mut SelectionManager,
    ) {
        egui::SidePanel::right("right_panel")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    for tab in RightTab::ALL {
                        ui.selectable_value(&mut self.active_tab, tab, tab.label());
                    }
                });
                ui.separator();

                match self.active_tab {
                    RightTab::Scene => self.scene.show(ui, document, selection),
                    other => {
                        ui.add_space(16.0);
                        ui.vertical_centered(|ui| {
                            ui.weak(format!("{} — coming soon", other.label()));
                        });
                    }
                }
            });
    }
}
