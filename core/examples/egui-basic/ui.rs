use cgmath::{InnerSpace, Vector3};
use wgpu_engine::common::RgbaColor;
use wgpu_engine::operator::BuiltinOperatorId;
use wgpu_engine::scene::{EffectiveVisibility, Light, LightType, NodeId, Visibility, MAX_LIGHTS};
use wgpu_engine::Viewer;

const WALK_OPERATOR_ID: u32 = BuiltinOperatorId::Walk as u32;
const NAV_OPERATOR_ID: u32 = BuiltinOperatorId::Navigation as u32;

/// Tab selection for the left panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum LeftPanelTab {
    #[default]
    Scene,
    Lights,
    Environment,
}

/// A visibility change requested by the UI.
pub struct VisibilityChange {
    pub node_id: NodeId,
    pub new_visibility: Visibility,
}

/// Actions requested by the UI that need to be handled by the application.
#[derive(Default)]
pub struct UiActions {
    pub load_file: bool,
    pub clear_scene: bool,
    pub add_light: Option<LightType>,
    pub load_environment: bool,
    pub clear_environment: bool,
    pub visibility_changes: Vec<VisibilityChange>,
}

/// Build all egui UI panels and return any actions requested.
pub fn build(ctx: &egui::Context, viewer: &mut Viewer) -> UiActions {
    let mut actions = UiActions::default();

    let mode_info = get_mode_info(viewer);

    build_performance_panel(ctx, &mode_info, viewer);
    build_left_panel(ctx, viewer, &mut actions);
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

/// Left panel with tabs for scene and lights.
fn build_left_panel(ctx: &egui::Context, viewer: &mut Viewer, actions: &mut UiActions) {
    egui::SidePanel::new(egui::panel::Side::Left, "Left Panel")
        .default_width(220.0)
        .show(ctx, |ui| {
            // Get/set tab state from egui memory
            let tab_id = ui.id().with("left_panel_tab");
            let mut current_tab = ui.memory(|mem| {
                mem.data.get_temp::<LeftPanelTab>(tab_id).unwrap_or_default()
            });

            // Tab selector
            ui.horizontal(|ui| {
                ui.selectable_value(&mut current_tab, LeftPanelTab::Scene, "Scene");
                ui.selectable_value(&mut current_tab, LeftPanelTab::Lights, "Lights");
                ui.selectable_value(&mut current_tab, LeftPanelTab::Environment, "Env");
            });

            // Store updated tab
            ui.memory_mut(|mem| mem.data.insert_temp(tab_id, current_tab));

            ui.separator();

            // Tab content
            match current_tab {
                LeftPanelTab::Scene => build_scene_tab(ui, viewer, actions),
                LeftPanelTab::Lights => build_lights_tab(ui, viewer, actions),
                LeftPanelTab::Environment => build_environment_tab(ui, viewer, actions),
            }
        });
}

/// Scene tab content with load/clear buttons and tree view.
fn build_scene_tab(ui: &mut egui::Ui, viewer: &Viewer, actions: &mut UiActions) {
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
                    render_node_tree(ui, viewer.scene(), root_id, 0, actions);
                }
            }
        });
}

/// Lights tab content with add/edit/delete controls.
fn build_lights_tab(ui: &mut egui::Ui, viewer: &mut Viewer, actions: &mut UiActions) {
    // Add light controls
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

    let light_count = viewer.scene().lights.len();
    ui.label(format!("({}/{} lights)", light_count, MAX_LIGHTS));

    ui.separator();

    // Light list with editors
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if light_count == 0 {
                ui.label("No lights in scene");
            } else {
                let mut light_to_delete: Option<usize> = None;

                for i in 0..viewer.scene().lights.len() {
                    let delete_requested = build_light_editor(ui, viewer, i);
                    if delete_requested {
                        light_to_delete = Some(i);
                    }
                    ui.separator();
                }

                // Handle deletion after iteration
                if let Some(idx) = light_to_delete {
                    viewer.scene_mut().lights.remove(idx);
                }
            }
        });
}

/// Environment tab content with HDR loading controls.
fn build_environment_tab(ui: &mut egui::Ui, viewer: &Viewer, actions: &mut UiActions) {
    ui.horizontal(|ui| {
        if ui.button("Load HDR...").clicked() {
            actions.load_environment = true;
        }
        if ui.button("Clear").clicked() {
            actions.clear_environment = true;
        }
    });

    ui.separator();

    // Show current environment status
    let scene = viewer.scene();
    if let Some(env_id) = scene.active_environment_map {
        if let Some(env_map) = scene.environment_maps.get(&env_id) {
            ui.label(format!("Active: Environment #{}", env_id));
            ui.label(format!("Intensity: {:.2}", env_map.intensity()));
            ui.label(format!("Rotation: {:.1}°", env_map.rotation().to_degrees()));
            if env_map.needs_generation() {
                ui.label("Status: Pending generation");
            } else {
                ui.label("Status: Ready");
            }
        }
    } else {
        ui.label("No environment map active");
        ui.label("");
        ui.label("Load an HDR file to enable");
        ui.label("image-based lighting (IBL)");
    }
}

/// Build editor UI for a single light. Returns true if delete was requested.
fn build_light_editor(ui: &mut egui::Ui, viewer: &mut Viewer, index: usize) -> bool {
    let mut delete_requested = false;

    // Get light info for the header
    let light_type_name = match &viewer.scene().lights[index] {
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
            // Edit the light properties
            let light = &mut viewer.scene_mut().lights[index];

            match light {
                Light::Point {
                    position,
                    color,
                    intensity,
                    range,
                } => {
                    build_color_edit(ui, color);
                    build_intensity_edit(ui, intensity);
                    build_position_edit(ui, position);
                    build_range_edit(ui, range);
                }
                Light::Directional {
                    direction,
                    color,
                    intensity,
                } => {
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
            // Ensure inner <= outer
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
            // Ensure outer >= inner
            if *outer < *inner {
                *inner = *outer;
            }
        }
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
    actions: &mut UiActions,
) {
    let Some(node) = scene.get_node(node_id) else {
        return;
    };

    let has_children = !node.children().is_empty();
    let has_instance = node.instance().is_some();

    // Get visibility state
    let visibility = node.visibility();
    let effective_visibility = scene.node_effective_visibility(node_id);
    let mut is_visible = visibility == Visibility::Visible;
    let is_indeterminate = effective_visibility == EffectiveVisibility::Mixed;

    // Build node label
    let label = if let Some(ref name) = node.name {
        name.clone()
    } else if has_instance {
        format!("Instance #{}", node_id)
    } else {
        format!("Node #{}", node_id)
    };

    let icon = if has_instance || has_children { "+" } else { "-" };
    let display_label = format!("{} {}", icon, label);

    // Dim text for invisible nodes
    let text_alpha = if effective_visibility == EffectiveVisibility::Invisible {
        0.5
    } else {
        1.0
    };

    // Clone children before ui.horizontal to avoid borrow issues
    let children: Vec<NodeId> = node.children().to_vec();

    if has_children {
        let id = ui.make_persistent_id(format!("node_{}", node_id));
        let mut state =
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, depth < 2);

        // Header row: checkbox + toggle + label
        ui.horizontal(|ui| {
            // Visibility checkbox
            let checkbox =
                egui::Checkbox::without_text(&mut is_visible).indeterminate(is_indeterminate);
            if ui.add(checkbox).changed() {
                let new_visibility = if is_visible {
                    Visibility::Visible
                } else {
                    Visibility::Invisible
                };
                actions.visibility_changes.push(VisibilityChange {
                    node_id,
                    new_visibility,
                });
            }

            // Collapse toggle button
            state.show_toggle_button(ui, egui::collapsing_header::paint_default_icon);

            // Label
            let text_color = ui.visuals().text_color().gamma_multiply(text_alpha);
            ui.colored_label(text_color, &display_label);
        });

        // Body (outside horizontal so indentation works)
        state.show_body_unindented(ui, |ui| {
            ui.indent(id, |ui| {
                for &child_id in &children {
                    render_node_tree(ui, scene, child_id, depth + 1, actions);
                }
            });
        });
    } else {
        // Leaf node: just checkbox + label
        ui.horizontal(|ui| {
            let checkbox =
                egui::Checkbox::without_text(&mut is_visible).indeterminate(is_indeterminate);
            if ui.add(checkbox).changed() {
                let new_visibility = if is_visible {
                    Visibility::Visible
                } else {
                    Visibility::Invisible
                };
                actions.visibility_changes.push(VisibilityChange {
                    node_id,
                    new_visibility,
                });
            }

            let text_color = ui.visuals().text_color().gamma_multiply(text_alpha);
            ui.colored_label(text_color, &display_label);
        });
    }
}
