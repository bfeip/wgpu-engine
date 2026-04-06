use duck_engine_viewer::operator::NavigationMode;
use duck_engine_viewer::Viewer;

use super::ModeInfo;

pub fn show(ui: &mut egui::Ui, viewer: &Viewer, mode: &ModeInfo) {
    build_camera_section(ui, viewer);
    ui.separator();
    build_controls_section(ui, mode);
    ui.separator();
    build_operators_section(ui, viewer);
    ui.separator();
    build_selection_section(ui, viewer);
    ui.separator();
    build_scene_info_section(ui, viewer);
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

    match mode.mode {
        NavigationMode::Walk => {
            ui.label("WASD: Move");
            ui.label("Left Mouse Drag: Look around");
        }
        NavigationMode::Orbit => {
            ui.label("Left Mouse Drag: Orbit camera");
            ui.label("Right Mouse Drag: Pan camera");
            ui.label("Mouse Wheel: Zoom in/out");
        }
    }

    ui.separator();
    ui.label("C: Cycle mode");
    ui.label("O: Toggle ortho/perspective");
    ui.label("ESC: Exit application");
}

fn build_operators_section(ui: &mut egui::Ui, viewer: &Viewer) {
    ui.heading("Operators");

    for op in viewer.operator_manager().iter() {
        ui.label(format!("  {}", op.name()));
    }
}

fn build_selection_section(ui: &mut egui::Ui, viewer: &Viewer) {
    ui.heading("Selection");

    let selection = viewer.selection();

    if selection.is_empty() {
        ui.label("(none)");
    } else {
        ui.label(format!("Count: {}", selection.len()));

        if let Some(primary) = selection.primary() {
            let node_id = primary.node_id();
            let node_label = viewer
                .scene()
                .get_node(node_id)
                .and_then(|n| n.name.clone())
                .unwrap_or_else(|| format!("Node #{}", node_id));
            ui.label(format!("Primary: {}", node_label));
        }

        ui.label("Selected:");
        for item in selection.iter() {
            let node_id = item.node_id();
            let node_label = viewer
                .scene()
                .get_node(node_id)
                .and_then(|n| n.name.clone())
                .unwrap_or_else(|| format!("Node #{}", node_id));
            ui.label(format!("  • {}", node_label));
        }
    }
}

fn build_scene_info_section(ui: &mut egui::Ui, viewer: &Viewer) {
    ui.heading("Scene Info");
    ui.label(format!("Meshes: {}", viewer.scene().mesh_count()));
    ui.label(format!("Instances: {}", viewer.scene().instance_count()));
    ui.label(format!("Nodes: {}", viewer.scene().node_count()));
    ui.label(format!("Lights: {}", viewer.scene().lights().len()));
}
