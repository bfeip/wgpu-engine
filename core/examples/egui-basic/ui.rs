use wgpu_engine::Viewer;
use wgpu_engine::operator::BuiltinOperatorId;
use wgpu_engine::scene::NodeId;

const WALK_OPERATOR_ID: u32 = BuiltinOperatorId::Walk as u32;
const NAV_OPERATOR_ID: u32 = BuiltinOperatorId::Navigation as u32;

/// Actions requested by the UI that need to be handled by the application.
#[derive(Default)]
pub struct UiActions {
    pub load_file: bool,
    pub clear_scene: bool,
}

/// Build all egui UI panels and return any actions requested.
pub fn build(ctx: &egui::Context, viewer: &Viewer) -> UiActions {
    let mut actions = UiActions::default();

    let mode_info = get_mode_info(viewer);

    build_performance_panel(ctx, &mode_info, viewer);
    build_scene_panel(ctx, viewer, &mut actions);
    build_info_panel(ctx, viewer, &mode_info);

    actions
}

/// Information about the current navigation mode.
struct ModeInfo {
    is_walk_mode: bool,
    is_nav_mode: bool,
}

fn get_mode_info(viewer: &Viewer) -> ModeInfo {
    let walk_pos = viewer.operator_manager().position(WALK_OPERATOR_ID);
    let nav_pos = viewer.operator_manager().position(NAV_OPERATOR_ID);

    let is_walk_mode = match (walk_pos, nav_pos) {
        (Some(w), Some(n)) => w < n,
        (Some(_), None) => true,
        _ => false,
    };
    let is_nav_mode = !is_walk_mode && nav_pos.is_some();

    ModeInfo {
        is_walk_mode,
        is_nav_mode,
    }
}

/// Top panel showing FPS and current mode.
fn build_performance_panel(ctx: &egui::Context, mode: &ModeInfo, viewer: &Viewer) {
    egui::TopBottomPanel::new(egui::panel::TopBottomSide::Top, "Performance")
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "FPS: {:.1}",
                    ctx.input(|i| i.stable_dt).recip()
                ));
                ui.separator();

                if mode.is_walk_mode {
                    ui.label("Mode: Walk");
                } else if mode.is_nav_mode {
                    ui.label("Mode: Orbit");
                } else if let Some(front) = viewer.operator_manager().front() {
                    ui.label(format!("Mode: {}", front.name()));
                }
            });
        });
}

/// Left panel with scene controls and tree view.
fn build_scene_panel(ctx: &egui::Context, viewer: &Viewer, actions: &mut UiActions) {
    egui::SidePanel::new(egui::panel::Side::Left, "Scene Controls")
        .default_width(200.0)
        .show(ctx, |ui| {
            ui.heading("Scene");

            ui.horizontal(|ui| {
                if ui.button("Load glTF...").clicked() {
                    actions.load_file = true;
                }
                if ui.button("Clear").clicked() {
                    actions.clear_scene = true;
                }
            });

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

/// Right panel with camera info, controls, and scene statistics.
fn build_info_panel(ctx: &egui::Context, viewer: &Viewer, mode: &ModeInfo) {
    egui::SidePanel::new(egui::panel::Side::Right, "Viewer Info")
        .default_width(200.0)
        .show(ctx, |ui| {
            build_camera_section(ui, viewer);
            ui.separator();
            build_controls_section(ui, mode);
            ui.separator();
            build_operators_section(ui, viewer, mode);
            ui.separator();
            build_scene_info_section(ui, viewer);
        });
}

fn build_camera_section(ui: &mut egui::Ui, viewer: &Viewer) {
    ui.heading("Camera");

    let camera = viewer.camera();
    ui.label(format!(
        "Projection: {}",
        if camera.ortho { "Orthographic" } else { "Perspective" }
    ));
    ui.label(format!(
        "Position: ({:.2}, {:.2}, {:.2})",
        camera.eye.x, camera.eye.y, camera.eye.z
    ));
    ui.label(format!(
        "Target: ({:.2}, {:.2}, {:.2})",
        camera.target.x, camera.target.y, camera.target.z
    ));
}

fn build_controls_section(ui: &mut egui::Ui, mode: &ModeInfo) {
    ui.heading("Controls");

    if mode.is_walk_mode {
        ui.label("WASD: Move");
        ui.label("Left Mouse Drag: Look around");
    } else if mode.is_nav_mode {
        ui.label("Left Mouse Drag: Orbit camera");
        ui.label("Right Mouse Drag: Pan camera");
        ui.label("Mouse Wheel: Zoom in/out");
    } else {
        ui.label("WASD: Walk movement");
        ui.label("Left Mouse Drag: Look around / Orbit");
        ui.label("Right Mouse Drag: Pan camera");
        ui.label("Mouse Wheel: Zoom in/out");
    }

    ui.separator();
    ui.label("C: Cycle mode");
    ui.label("O: Toggle ortho/perspective");
    ui.label("ESC: Exit application");
}

fn build_operators_section(ui: &mut egui::Ui, viewer: &Viewer, mode: &ModeInfo) {
    ui.heading("Operators");

    let active_nav_id = if mode.is_walk_mode {
        Some(WALK_OPERATOR_ID)
    } else if mode.is_nav_mode {
        Some(NAV_OPERATOR_ID)
    } else {
        None
    };

    for op in viewer.operator_manager().iter() {
        let prefix = if Some(op.id()) == active_nav_id { "> " } else { "  " };
        ui.label(format!("{}{}", prefix, op.name()));
    }
}

fn build_scene_info_section(ui: &mut egui::Ui, viewer: &Viewer) {
    ui.heading("Scene Info");
    ui.label(format!("Meshes: {}", viewer.scene().meshes.len()));
    ui.label(format!("Instances: {}", viewer.scene().instances.len()));
    ui.label(format!("Nodes: {}", viewer.scene().nodes.len()));
    ui.label(format!("Lights: {}", viewer.scene().lights.len()));
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
