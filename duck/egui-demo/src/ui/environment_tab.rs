use duck_engine_viewer::Viewer;

use super::UiActions;

pub fn show(ui: &mut egui::Ui, viewer: &mut Viewer, actions: &mut UiActions) {
    ui.horizontal(|ui| {
        if ui.button("Load HDR...").clicked() {
            actions.load_environment = true;
        }
        if ui.button("Clear").clicked() {
            actions.clear_environment = true;
        }
    });

    ui.separator();

    let scene_arc = viewer.scene();
    let env_id = scene_arc.lock().unwrap().active_environment_map();
    if let Some(env_id) = env_id {
        let (mut intensity, rotation_deg) = {
            let scene = scene_arc.lock().unwrap();
            let intensity = scene
                .get_environment_map(env_id)
                .map_or(1.0, |e| e.intensity());
            let rotation_deg = scene
                .get_environment_map(env_id)
                .map_or(0.0, |e| e.rotation().to_degrees());
            (intensity, rotation_deg)
        };

        ui.label(format!("Active: Environment #{}", env_id));
        if ui
            .add(egui::Slider::new(&mut intensity, 0.0..=5.0).text("Intensity"))
            .changed()
            && let Some(env_map) = scene_arc.lock().unwrap().get_environment_map_mut(env_id) {
                env_map.set_intensity(intensity);
            }
        ui.label(format!("Rotation: {:.1}°", rotation_deg));
    } else {
        ui.label("No environment map active");
        ui.label("");
        ui.label("Load an HDR file to enable");
        ui.label("image-based lighting (IBL)");
    }
}
