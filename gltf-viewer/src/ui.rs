use wgpu_engine::Viewer;
use wgpu_engine::scene::NodeId;

/// Actions requested by the UI that need to be handled by the application.
#[derive(Default)]
pub struct UiActions {
    pub load_file: bool,
    pub clear_scene: bool,
}

/// Build all egui UI panels and return any actions requested.
pub fn build(ctx: &egui::Context, viewer: &Viewer) -> UiActions {
    let mut actions = UiActions::default();

    build_scene_panel(ctx, viewer, &mut actions);

    actions
}

/// Left panel with scene controls and tree view.
fn build_scene_panel(ctx: &egui::Context, viewer: &Viewer, actions: &mut UiActions) {
    egui::SidePanel::new(egui::panel::Side::Left, "Scene Controls")
        .default_width(200.0)
        .show(ctx, |ui| {
            ui.heading("Scene");

            ui.horizontal(|ui| {
                #[cfg(not(target_arch = "wasm32"))]
                let load_text = "Load glTF...";
                #[cfg(target_arch = "wasm32")]
                let load_text = "Choose File...";

                if ui.button(load_text).clicked() {
                    actions.load_file = true;
                }
                if ui.button("Clear").clicked() {
                    actions.clear_scene = true;
                }
            });

            #[cfg(target_arch = "wasm32")]
            {
                ui.small("(Drag & drop also supported)");
            }

            ui.separator();

            ui.heading("Scene Tree");

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if viewer.scene().root_nodes.is_empty() {
                        ui.label("(empty)");
                    } else {
                        for &root_id in &viewer.scene().root_nodes {
                            render_node_tree(ui, viewer.scene(), root_id, 0);
                        }
                    }
                });
        });
}

/// Recursively render a node and its children in the scene tree.
fn render_node_tree(
    ui: &mut egui::Ui,
    scene: &wgpu_engine::scene::Scene,
    node_id: NodeId,
    depth: usize,
) {
    let Some(node) = scene.get_node(node_id) else {
        return;
    };

    let has_children = !node.children().is_empty();
    let has_instance = node.instance().is_some();

    let label = if let Some(ref name) = node.name {
        name.clone()
    } else if has_instance {
        format!("Instance #{}", node_id)
    } else {
        format!("Node #{}", node_id)
    };

    let icon = if has_instance || has_children { "+" } else { "-" };
    let display_label = format!("{} {}", icon, label);

    if has_children {
        let id = ui.make_persistent_id(format!("node_{}", node_id));
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, depth < 2)
            .show_header(ui, |ui| {
                ui.label(&display_label);
            })
            .body(|ui| {
                for &child_id in node.children() {
                    render_node_tree(ui, scene, child_id, depth + 1);
                }
            });
    } else {
        ui.label(&display_label);
    }
}
