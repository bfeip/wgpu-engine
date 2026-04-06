use cgmath::{InnerSpace, Vector3};
use wgpu_engine_viewer::common::RgbaColor;
use wgpu_engine_viewer::scene::{Light, LightType, MAX_LIGHTS};
use wgpu_engine_viewer::Viewer;

use super::UiActions;

pub fn show(ui: &mut egui::Ui, viewer: &mut Viewer, actions: &mut UiActions) {
    ui.horizontal(|ui| {
        ui.label("Add:");
        if ui.button("Point").clicked() {
            actions.add_light = Some(LightType::Point);
        }
        if ui.button("Dir").clicked() {
            actions.add_light = Some(LightType::Directional);
        }
        if ui.button("Spot").clicked() {
            actions.add_light = Some(LightType::Spot);
        }
    });

    let light_count = viewer.scene().lights().len();
    ui.label(format!("({}/{} lights)", light_count, MAX_LIGHTS));

    ui.separator();

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if light_count == 0 {
                ui.label("No lights in scene");
            } else {
                let mut light_to_delete: Option<usize> = None;

                for i in 0..viewer.scene().lights().len() {
                    let delete_requested = build_light_editor(ui, viewer, i);
                    if delete_requested {
                        light_to_delete = Some(i);
                    }
                    ui.separator();
                }

                if let Some(idx) = light_to_delete {
                    viewer.scene_mut().remove_light(idx);
                }
            }
        });
}

/// Build editor UI for a single light. Returns true if delete was requested.
fn build_light_editor(ui: &mut egui::Ui, viewer: &mut Viewer, index: usize) -> bool {
    let mut delete_requested = false;

    let light_type_name = match &viewer.scene().lights()[index] {
        Light::Point { .. } => "Point",
        Light::Directional { .. } => "Directional",
        Light::Spot { .. } => "Spot",
    };

    let header_id = ui.make_persistent_id(format!("light_{}", index));

    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), header_id, true)
        .show_header(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("{} #{}", light_type_name, index));
                if ui.small_button("X").clicked() {
                    delete_requested = true;
                }
            });
        })
        .body(|ui| {
            let light = &mut viewer.scene_mut().lights_mut()[index];

            match light {
                Light::Point { position, color, intensity, range, .. } => {
                    build_color_edit(ui, color);
                    build_intensity_edit(ui, intensity);
                    build_position_edit(ui, position);
                    build_range_edit(ui, range);
                }
                Light::Directional { direction, color, intensity, .. } => {
                    build_color_edit(ui, color);
                    build_intensity_edit(ui, intensity);
                    build_direction_edit(ui, direction);
                }
                Light::Spot {
                    position,
                    direction,
                    color,
                    intensity,
                    range,
                    inner_cone_angle,
                    outer_cone_angle,
                    ..
                } => {
                    build_color_edit(ui, color);
                    build_intensity_edit(ui, intensity);
                    build_position_edit(ui, position);
                    build_direction_edit(ui, direction);
                    build_range_edit(ui, range);
                    build_cone_angles_edit(ui, inner_cone_angle, outer_cone_angle);
                }
            }
        });

    delete_requested
}

fn build_color_edit(ui: &mut egui::Ui, color: &mut RgbaColor) {
    ui.horizontal(|ui| {
        ui.label("Color:");
        let mut rgb = [color.r, color.g, color.b];
        if ui.color_edit_button_rgb(&mut rgb).changed() {
            color.r = rgb[0];
            color.g = rgb[1];
            color.b = rgb[2];
        }
    });
}

fn build_intensity_edit(ui: &mut egui::Ui, intensity: &mut f32) {
    ui.horizontal(|ui| {
        ui.label("Intensity:");
        ui.add(egui::DragValue::new(intensity).speed(0.1).range(0.0..=100.0));
    });
}

fn build_position_edit(ui: &mut egui::Ui, position: &mut Vector3<f32>) {
    ui.label("Position:");
    ui.horizontal(|ui| {
        ui.label("X:");
        ui.add(egui::DragValue::new(&mut position.x).speed(0.1));
        ui.label("Y:");
        ui.add(egui::DragValue::new(&mut position.y).speed(0.1));
    });
    ui.horizontal(|ui| {
        ui.label("Z:");
        ui.add(egui::DragValue::new(&mut position.z).speed(0.1));
    });
}

fn build_direction_edit(ui: &mut egui::Ui, direction: &mut Vector3<f32>) {
    ui.label("Direction:");
    ui.horizontal(|ui| {
        ui.label("X:");
        ui.add(egui::DragValue::new(&mut direction.x).speed(0.01).range(-1.0..=1.0));
        ui.label("Y:");
        ui.add(egui::DragValue::new(&mut direction.y).speed(0.01).range(-1.0..=1.0));
    });
    ui.horizontal(|ui| {
        ui.label("Z:");
        ui.add(egui::DragValue::new(&mut direction.z).speed(0.01).range(-1.0..=1.0));
        if ui.button("Norm").clicked() && direction.magnitude() > 0.0 {
            *direction = direction.normalize();
        }
    });
}

fn build_range_edit(ui: &mut egui::Ui, range: &mut f32) {
    ui.horizontal(|ui| {
        ui.label("Range:");
        ui.add(egui::DragValue::new(range).speed(0.1).range(0.0..=1000.0));
        if *range == 0.0 {
            ui.label("(infinite)");
        }
    });
}

fn build_cone_angles_edit(ui: &mut egui::Ui, inner: &mut f32, outer: &mut f32) {
    let mut inner_deg = inner.to_degrees();
    let mut outer_deg = outer.to_degrees();

    ui.horizontal(|ui| {
        ui.label("Inner cone:");
        if ui
            .add(egui::DragValue::new(&mut inner_deg).speed(1.0).range(0.0..=90.0).suffix("°"))
            .changed()
        {
            *inner = inner_deg.to_radians();
            if *inner > *outer {
                *outer = *inner;
            }
        }
    });

    ui.horizontal(|ui| {
        ui.label("Outer cone:");
        if ui
            .add(egui::DragValue::new(&mut outer_deg).speed(1.0).range(0.0..=90.0).suffix("°"))
            .changed()
        {
            *outer = outer_deg.to_radians();
            if *outer < *inner {
                *inner = *outer;
            }
        }
    });
}
