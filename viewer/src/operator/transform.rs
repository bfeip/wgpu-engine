//! Transform operator for Blender-style grab/rotate/scale operations.
//!
//! Controls:
//! - G: Start grab/translate mode
//! - R: Start rotate mode
//! - S: Start scale mode
//! - H: Cycle gizmo handles (None → Translate → Rotate → Scale → None)
//! - X/Y/Z: Constrain to axis (toggle: none → world → local → none)
//! - Left-click / Enter: Confirm transform
//! - Right-click / Escape: Cancel transform
//! - Mouse movement: Adjust transform magnitude
//!
//! Two independent interaction flows:
//! 1. **Keyboard**: G/R/S activates a transform driven by mouse motion.
//! 2. **Gizmo drag**: H shows persistent handles; drag a handle for an
//!    axis-constrained transform that confirms on mouse release.

use std::cell::RefCell;
use std::rc::Rc;

use cgmath::{
    EuclideanSpace, InnerSpace, Matrix4, Point3, Quaternion, Rotation, SquareMatrix, Vector3,
};
use wgpu_engine_scene::common;

use crate::common::{
    apply_scale, centroid_of_slice, compose_rotation, decompose_matrix, local_axis_x, local_axis_y,
    local_axis_z, quaternion_from_axis_angle_safe, rotate_position_about_pivot,
    scale_position_about_pivot_local, scale_position_about_pivot_world, RgbaColor, Transform,
};
use crate::common::Axis;
use crate::event::{CallbackId, Event, EventContext, EventDispatcher, EventKind};
use crate::geom_query::pick_all_from_ray;
use crate::input::{ElementState, Key, MouseButton, NamedKey};
use crate::operator::{Operator, OperatorId};
use crate::scene::annotation::AnnotationId;
use crate::scene::gizmo::{self, GizmoType};
use crate::scene::{MaterialId, MeshId, NodeId};
use crate::scene_scale;

/// The type of transform being performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformMode {
    Translate,
    Rotate,
    Scale,
}

/// Axis constraint for the transform operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisConstraint {
    /// No constraint - free transform
    None,
    /// Constrain to world X axis
    WorldX,
    /// Constrain to world Y axis
    WorldY,
    /// Constrain to world Z axis
    WorldZ,
    /// Constrain to local X axis (relative to primary selection)
    LocalX,
    /// Constrain to local Y axis (relative to primary selection)
    LocalY,
    /// Constrain to local Z axis (relative to primary selection)
    LocalZ,
}

impl AxisConstraint {
    /// Returns the color for visual feedback (RGB = XYZ convention).
    fn color(&self) -> Option<RgbaColor> {
        match self {
            AxisConstraint::None => None,
            AxisConstraint::WorldX | AxisConstraint::LocalX => Some(RgbaColor::RED),
            AxisConstraint::WorldY | AxisConstraint::LocalY => Some(RgbaColor::GREEN),
            AxisConstraint::WorldZ | AxisConstraint::LocalZ => Some(RgbaColor::BLUE),
        }
    }

    /// Returns whether this is a local axis constraint.
    fn is_local(&self) -> bool {
        matches!(
            self,
            AxisConstraint::LocalX | AxisConstraint::LocalY | AxisConstraint::LocalZ
        )
    }
}

/// Tracks gizmo scene resources for cleanup.
///
/// Gizmo nodes are parented under the annotation root node so they are grouped
/// with other overlay content and hidden when annotations are toggled off.
///
// TODO: Once we support multiple overlay types beyond annotations and gizmos,
// consider introducing a dedicated overlay system with its own root node
// rather than piggy-backing on the annotation root.
struct GizmoState {
    /// Node IDs of the gizmo handles (one per axis: X, Y, Z).
    node_ids: Vec<NodeId>,
    /// Mesh IDs added to the scene for gizmo geometry.
    mesh_ids: Vec<MeshId>,
    /// Material IDs added to the scene for gizmo handles.
    material_ids: Vec<MaterialId>,
    /// Which axis is currently highlighted (hovered or active).
    highlighted_axis: Option<Axis>,
    /// Current gizmo type being displayed.
    current_type: Option<GizmoType>,
}

impl GizmoState {
    fn new() -> Self {
        Self {
            node_ids: Vec::new(),
            mesh_ids: Vec::new(),
            material_ids: Vec::new(),
            highlighted_axis: None,
            current_type: None,
        }
    }

    fn has_gizmo(&self) -> bool {
        self.current_type.is_some()
    }

    /// Build and add gizmo handles to the scene at the given pivot point.
    ///
    /// Handles are parented under the annotation root so they inherit annotation
    /// visibility and stay grouped with other overlay geometry.
    fn show(&mut self, gizmo_type: GizmoType, pivot: Point3<f32>, size: f32, ctx: &mut EventContext) {
        self.hide(ctx);

        let handles = gizmo::build_handles(gizmo_type, size);
        let annotation_root = ctx.scene.ensure_annotation_root();
        let pivot_transform = common::Transform::from_position(pivot);

        for handle in handles {
            let mesh_id = ctx.scene.add_mesh(handle.mesh);
            let material_id = ctx.scene.add_material(handle.material);
            let node_id = ctx
                .scene
                .add_instance_node(
                    Some(annotation_root),
                    mesh_id,
                    material_id,
                    None,
                    pivot_transform,
                )
                .expect("Failed to add gizmo node");

            self.node_ids.push(node_id);
            self.mesh_ids.push(mesh_id);
            self.material_ids.push(material_id);
        }

        self.current_type = Some(gizmo_type);
    }

    /// Remove all gizmo geometry from the scene.
    fn hide(&mut self, ctx: &mut EventContext) {
        for &node_id in &self.node_ids {
            ctx.scene.remove_node(node_id);
        }
        for &mesh_id in &self.mesh_ids {
            ctx.scene.remove_mesh(mesh_id);
        }
        for &material_id in &self.material_ids {
            ctx.scene.remove_material(material_id);
        }

        self.node_ids.clear();
        self.mesh_ids.clear();
        self.material_ids.clear();
        self.highlighted_axis = None;
        self.current_type = None;
    }

    /// Update the gizmo position (e.g. when pivot changes).
    fn update_position(&self, pivot: Point3<f32>, ctx: &mut EventContext) {
        for &node_id in &self.node_ids {
            if ctx.scene.has_node(node_id) {
                ctx.scene.set_node_position(node_id, pivot);
            }
        }
    }

    /// Pick which gizmo handle (if any) is under the given screen position.
    fn pick_handle(&self, cursor_x: f32, cursor_y: f32, ctx: &EventContext) -> Option<Axis> {
        if !self.has_gizmo() {
            return None;
        }

        let ray = ctx.camera.ray_from_screen_point(cursor_x, cursor_y, ctx.size.0, ctx.size.1);
        let results = pick_all_from_ray(&ray, ctx.scene);

        // Find the first hit that matches a gizmo node
        for result in &results {
            for (i, &node_id) in self.node_ids.iter().enumerate() {
                if result.node_id == node_id {
                    return Some(Axis::ALL[i]);
                }
            }
        }

        None
    }

    /// Highlight a specific axis handle (or clear highlight with None).
    fn set_highlight(&mut self, axis: Option<Axis>, ctx: &mut EventContext) {
        if self.highlighted_axis == axis {
            return;
        }

        // Restore previous highlight to normal color
        if let Some(prev_axis) = self.highlighted_axis {
            let idx = axis_index(prev_axis);
            if let Some(&mat_id) = self.material_ids.get(idx) {
                if let Some(mat) = ctx.scene.get_material_mut(mat_id) {
                    mat.set_base_color_factor(prev_axis.color());
                }
            }
        }

        // Apply highlight color to new axis
        if let Some(new_axis) = axis {
            let idx = axis_index(new_axis);
            if let Some(&mat_id) = self.material_ids.get(idx) {
                if let Some(mat) = ctx.scene.get_material_mut(mat_id) {
                    mat.set_base_color_factor(gizmo::highlight_color(new_axis));
                }
            }
        }

        self.highlighted_axis = axis;
    }
}

/// Maps an axis to its index in the gizmo handle arrays (X=0, Y=1, Z=2).
fn axis_index(axis: Axis) -> usize {
    match axis {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    }
}

/// Maps an axis constraint to the corresponding Axis for gizmo highlighting.
fn axis_from_constraint(constraint: &AxisConstraint) -> Option<Axis> {
    match constraint {
        AxisConstraint::WorldX | AxisConstraint::LocalX => Some(Axis::X),
        AxisConstraint::WorldY | AxisConstraint::LocalY => Some(Axis::Y),
        AxisConstraint::WorldZ | AxisConstraint::LocalZ => Some(Axis::Z),
        AxisConstraint::None => None,
    }
}

/// Original transform state for a node (used for cancel/restore).
#[derive(Debug, Clone)]
struct OriginalTransform {
    node_id: NodeId,
    local_transform: Transform,
    world_transform: Transform,
    parent_world_transform: Transform,
}

/// Internal state for the transform operator.
struct TransformState {
    /// Current transform mode (None when inactive).
    mode: Option<TransformMode>,

    /// Current axis constraint.
    axis_constraint: AxisConstraint,

    /// Original transforms of selected nodes (for cancel/restore).
    original_transforms: Vec<OriginalTransform>,

    /// The rotation of the primary selected node (for local axis transforms).
    primary_rotation: Quaternion<f32>,

    /// Accumulated mouse movement since transform started.
    accumulated_delta: (f32, f32),

    /// Center point of selection in world space (pivot point for rotation/scale).
    pivot_world: Point3<f32>,

    /// Model radius for scaling sensitivity.
    model_radius: f32,

    /// Annotation IDs for visual feedback (cleaned up on finish).
    annotation_ids: Vec<AnnotationId>,

    /// Gizmo handle state (3D visual handles for the active transform).
    gizmo: GizmoState,

    /// Which gizmo type to display (persists across transforms, cycled by H key).
    /// Independent of `mode` — the gizmo is a persistent visual tool, while
    /// `mode` tracks the active keyboard-driven transform.
    gizmo_mode: Option<GizmoType>,
}

impl TransformState {
    fn new() -> Self {
        Self {
            mode: None,
            axis_constraint: AxisConstraint::None,
            original_transforms: Vec::new(),
            primary_rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            accumulated_delta: (0.0, 0.0),
            pivot_world: Point3::origin(),
            model_radius: 1.0,
            annotation_ids: Vec::new(),
            gizmo: GizmoState::new(),
            gizmo_mode: None,
        }
    }

    /// Returns true if a transform operation is currently active.
    fn is_active(&self) -> bool {
        self.mode.is_some()
    }

    /// Cycles the axis constraint for a given axis key.
    /// None → World → Local → None
    fn cycle_axis_constraint(&mut self, axis: char) {
        self.axis_constraint = match (axis, &self.axis_constraint) {
            // X axis cycling
            ('x', AxisConstraint::None) => AxisConstraint::WorldX,
            ('x', AxisConstraint::WorldX) => AxisConstraint::LocalX,
            ('x', AxisConstraint::LocalX) => AxisConstraint::None,
            ('x', _) => AxisConstraint::WorldX, // Switch from other axis

            // Y axis cycling
            ('y', AxisConstraint::None) => AxisConstraint::WorldY,
            ('y', AxisConstraint::WorldY) => AxisConstraint::LocalY,
            ('y', AxisConstraint::LocalY) => AxisConstraint::None,
            ('y', _) => AxisConstraint::WorldY, // Switch from other axis

            // Z axis cycling
            ('z', AxisConstraint::None) => AxisConstraint::WorldZ,
            ('z', AxisConstraint::WorldZ) => AxisConstraint::LocalZ,
            ('z', AxisConstraint::LocalZ) => AxisConstraint::None,
            ('z', _) => AxisConstraint::WorldZ, // Switch from other axis

            _ => self.axis_constraint,
        };
    }

    /// Get the constraint axis direction in world space.
    fn get_constraint_axis(&self) -> Option<Vector3<f32>> {
        match self.axis_constraint {
            AxisConstraint::None => None,
            AxisConstraint::WorldX => Some(Vector3::unit_x()),
            AxisConstraint::WorldY => Some(Vector3::unit_y()),
            AxisConstraint::WorldZ => Some(Vector3::unit_z()),
            AxisConstraint::LocalX => Some(local_axis_x(self.primary_rotation)),
            AxisConstraint::LocalY => Some(local_axis_y(self.primary_rotation)),
            AxisConstraint::LocalZ => Some(local_axis_z(self.primary_rotation)),
        }
    }

    /// Compute the translation delta based on mouse movement and constraints.
    fn compute_translation(&self, ctx: &EventContext) -> Vector3<f32> {
        let camera = &ctx.camera;
        let pivot = &self.pivot_world;
        let (width, height) = ctx.size;
        let (dx, dy) = self.accumulated_delta;

        let movement_plane = common::Plane::from_point(camera.forward(), *pivot);
        let Point3 { x: screen_x, y: screen_y, .. } = camera.project_point_screen(*pivot, width, height);
        let diff_ray = camera.ray_from_screen_point(screen_x + dx, screen_y + dy, width, height);
        let new_pivot = movement_plane.intersect_ray(&diff_ray)
            .map_or(pivot.clone(), |intersection| intersection.1);
        let move_vector = new_pivot - pivot;

        match self.get_constraint_axis() {
            None => {
                move_vector
            }
            Some(axis) => {
                axis * axis.dot(move_vector)
            }
        }
    }

    /// Compute the rotation based on mouse movement and constraints.
    fn compute_rotation(&self, ctx: &EventContext) -> Quaternion<f32> {
        // 0.5 degrees per pixel
        let sensitivity = 0.5_f32.to_radians();
        let angle = self.accumulated_delta.0 * sensitivity;

        let axis = match self.get_constraint_axis() {
            None => {
                // Free rotation: rotate around view axis
                ctx.camera.forward()
            }
            Some(axis) => axis,
        };

        quaternion_from_axis_angle_safe(axis, angle)
    }

    /// Compute the scale factor based on mouse movement and constraints.
    fn compute_scale(&self) -> Vector3<f32> {
        // 0.5% change per pixel
        let sensitivity = 0.005;
        let factor = 1.0 + self.accumulated_delta.0 * sensitivity;
        // Clamp to prevent negative or zero scale
        let factor = factor.max(0.01);

        match self.axis_constraint {
            AxisConstraint::None => Vector3::new(factor, factor, factor),
            AxisConstraint::WorldX | AxisConstraint::LocalX => Vector3::new(factor, 1.0, 1.0),
            AxisConstraint::WorldY | AxisConstraint::LocalY => Vector3::new(1.0, factor, 1.0),
            AxisConstraint::WorldZ | AxisConstraint::LocalZ => Vector3::new(1.0, 1.0, factor),
        }
    }

    /// Apply the current transform preview to all selected nodes.
    fn apply_preview_transform(&self, ctx: &mut EventContext) {
        let mode = match self.mode {
            Some(m) => m,
            None => return,
        };

        // Pre-compute transforms that need ctx (camera access)
        // This avoids borrow conflicts when we later mutate nodes
        let translation_delta = if mode == TransformMode::Translate {
            Some(self.compute_translation(ctx))
        } else {
            None
        };

        let rotation_quat = if mode == TransformMode::Rotate {
            Some(self.compute_rotation(ctx))
        } else {
            None
        };

        let scale_factor = if mode == TransformMode::Scale {
            Some(self.compute_scale())
        } else {
            None
        };

        // Now apply transforms to nodes
        for orig in &self.original_transforms {
            let inv_parent = orig
                .parent_world_transform
                .to_matrix()
                .invert()
                .unwrap_or(Matrix4::identity());

            if !ctx.scene.has_node(orig.node_id) {
                continue;
            }

            match mode {
                TransformMode::Translate => {
                    let delta = translation_delta.unwrap();
                    let new_world_pos = orig.world_transform.position + delta;
                    let new_local_pos =
                        Point3::from_homogeneous(inv_parent * new_world_pos.to_homogeneous());
                    ctx.scene.set_node_position(orig.node_id, new_local_pos);
                }
                TransformMode::Rotate => {
                    let rotation = rotation_quat.unwrap();

                    // Rotate world position around world pivot, convert to local
                    let new_world_pos = rotate_position_about_pivot(
                        orig.world_transform.position,
                        self.pivot_world,
                        rotation,
                    );
                    let new_local_pos =
                        Point3::from_homogeneous(inv_parent * new_world_pos.to_homogeneous());
                    ctx.scene.set_node_position(orig.node_id, new_local_pos);

                    // Convert world rotation to local space
                    let pr = orig.parent_world_transform.rotation;
                    let pr_inv = pr.conjugate();
                    let local_rotation = pr_inv * rotation * pr;
                    let new_rotation =
                        compose_rotation(orig.local_transform.rotation, local_rotation);
                    ctx.scene.set_node_rotation(orig.node_id, new_rotation);
                }
                TransformMode::Scale => {
                    let scale = scale_factor.unwrap();

                    if self.axis_constraint.is_local() {
                        // Local axis: scale in local space, but use world positions for pivot
                        let new_world_pos = scale_position_about_pivot_local(
                            orig.world_transform.position,
                            self.pivot_world,
                            scale,
                            self.primary_rotation,
                        );
                        let new_local_pos =
                            Point3::from_homogeneous(inv_parent * new_world_pos.to_homogeneous());
                        ctx.scene.set_node_position(orig.node_id, new_local_pos);
                        let new_scale = apply_scale(orig.local_transform.scale, scale);
                        ctx.scene.set_node_scale(orig.node_id, new_scale);
                    } else {
                        // World axis: scale world position around pivot, convert to local
                        let new_world_pos = scale_position_about_pivot_world(
                            orig.world_transform.position,
                            self.pivot_world,
                            scale,
                        );
                        let new_local_pos =
                            Point3::from_homogeneous(inv_parent * new_world_pos.to_homogeneous());
                        ctx.scene.set_node_position(orig.node_id, new_local_pos);

                        // Convert world-axis scale to local space
                        let pr_inv = orig.parent_world_transform.rotation.conjugate();
                        let local_scale = world_scale_to_local(scale, pr_inv);
                        let new_scale = apply_scale(orig.local_transform.scale, local_scale);
                        ctx.scene.set_node_scale(orig.node_id, new_scale);
                    }
                }
            }
        }
    }

    /// Restore all nodes to their original transforms.
    fn restore_original_transforms(&self, ctx: &mut EventContext) {
        for orig in &self.original_transforms {
            if ctx.scene.has_node(orig.node_id) {
                ctx.scene.set_node_transform(orig.node_id, orig.local_transform);
            }
        }
    }

    /// Update visual feedback annotations.
    fn update_visual_feedback(&mut self, ctx: &mut EventContext) {
        // Clear previous annotations
        for id in self.annotation_ids.drain(..) {
            ctx.scene.remove_annotation(id);
        }

        // Add axis constraint line if constrained
        if let Some(color) = self.axis_constraint.color() {
            if let Some(axis) = self.get_constraint_axis() {
                let half_length = self.model_radius * 2.0;
                let start = self.pivot_world - axis * half_length;
                let end = self.pivot_world + axis * half_length;
                let id = ctx.scene.annotations.add_line(start, end, color);
                self.annotation_ids.push(id);
            }
        }
    }

    /// Clean up annotations.
    fn cleanup_annotations(&mut self, ctx: &mut EventContext) {
        for id in self.annotation_ids.drain(..) {
            ctx.scene.remove_annotation(id);
        }
    }

    /// Show, reposition, or hide the gizmo based on `gizmo_mode` and selection.
    fn sync_gizmo(&mut self, ctx: &mut EventContext) {
        let selected = ctx.selection.selected_nodes();
        match (self.gizmo_mode, selected.is_empty()) {
            (Some(gizmo_type), false) => {
                let positions: Vec<Point3<f32>> = selected
                    .iter()
                    .map(|&nid| {
                        let m = ctx.scene.nodes_transform(nid);
                        Point3::new(m[3][0], m[3][1], m[3][2])
                    })
                    .collect();
                let pivot = centroid_of_slice(&positions).unwrap_or(Point3::origin());
                let model_radius =
                    scene_scale::model_radius_from_bounds(ctx.scene.bounding().as_ref());

                if self.gizmo.current_type == Some(gizmo_type) {
                    self.gizmo.update_position(pivot, ctx);
                } else {
                    self.gizmo.show(gizmo_type, pivot, model_radius * 0.15, ctx);
                }
            }
            _ => {
                self.gizmo.hide(ctx);
            }
        }
    }

    /// Reset state after transform completes (confirm or cancel).
    fn reset(&mut self) {
        self.mode = None;
        self.axis_constraint = AxisConstraint::None;
        self.original_transforms.clear();
        self.accumulated_delta = (0.0, 0.0);
    }
}

/// Converts a world-axis-constrained scale into the parent's local space.
///
/// For uniform scale (no constraint), returns as-is. For single-axis world
/// constraints, rotates the scale axis into local space and decomposes it
/// into per-axis scale contributions.
fn world_scale_to_local(scale: Vector3<f32>, parent_rotation_inv: Quaternion<f32>) -> Vector3<f32> {
    // For uniform scale (no constraint), return as-is
    if (scale.x - scale.y).abs() < 1e-6 && (scale.y - scale.z).abs() < 1e-6 {
        return scale;
    }
    // Find the world axis being scaled and its factor
    let (world_axis, factor) = if (scale.x - 1.0).abs() > 1e-6 {
        (Vector3::unit_x(), scale.x)
    } else if (scale.y - 1.0).abs() > 1e-6 {
        (Vector3::unit_y(), scale.y)
    } else {
        (Vector3::unit_z(), scale.z)
    };
    // Rotate to local space
    let local_axis = parent_rotation_inv.rotate_vector(world_axis).normalize();
    // Decompose into per-axis scale contributions
    let factor_minus_1 = factor - 1.0;
    Vector3::new(
        1.0 + local_axis.x.abs() * factor_minus_1,
        1.0 + local_axis.y.abs() * factor_minus_1,
        1.0 + local_axis.z.abs() * factor_minus_1,
    )
}

/// Operator for Blender-style transform operations (grab/rotate/scale).
pub struct TransformOperator {
    id: OperatorId,
    state: Rc<RefCell<TransformState>>,
    callback_ids: Vec<CallbackId>,
}

impl TransformOperator {
    /// Creates a new transform operator with the given ID.
    pub fn new(id: OperatorId) -> Self {
        Self {
            id,
            state: Rc::new(RefCell::new(TransformState::new())),
            callback_ids: Vec::new(),
        }
    }

    /// Start a transform operation with the given mode.
    fn start_transform(state: &mut TransformState, mode: TransformMode, ctx: &mut EventContext) {
        // Get selected nodes
        let selected_nodes = ctx.selection.selected_nodes();
        if selected_nodes.is_empty() {
            return;
        }

        // Store original transforms with world-space info
        let mut world_positions: Vec<Point3<f32>> = Vec::new();
        for node_id in &selected_nodes {
            if let Some(node) = ctx.scene.get_node(*node_id) {
                let world_matrix = ctx.scene.nodes_transform(*node_id);
                let world_transform = decompose_matrix(&world_matrix);

                let parent_world_transform = if let Some(parent_id) = node.parent() {
                    let parent_matrix = ctx.scene.nodes_transform(parent_id);
                    decompose_matrix(&parent_matrix)
                } else {
                    Transform::IDENTITY
                };

                world_positions.push(world_transform.position);
                state.original_transforms.push(OriginalTransform {
                    node_id: *node_id,
                    local_transform: node.transform(),
                    world_transform,
                    parent_world_transform,
                });
            }
        }

        if world_positions.is_empty() {
            return;
        }

        // Compute pivot as centroid of world-space positions
        state.pivot_world = centroid_of_slice(&world_positions).unwrap_or(Point3::origin());

        // Store primary selection's rotation for local axis transforms
        if let Some(primary) = ctx.selection.primary() {
            if let Some(node) = ctx.scene.get_node(primary.node_id()) {
                state.primary_rotation = node.rotation();
            }
        }

        // Get model radius for sensitivity scaling
        state.model_radius =
            scene_scale::model_radius_from_bounds(ctx.scene.bounding().as_ref());

        // Activate the transform
        state.mode = Some(mode);
        state.axis_constraint = AxisConstraint::None;
        state.accumulated_delta = (0.0, 0.0);
    }

    /// Confirm the transform (keep current state).
    ///
    /// The gizmo remains visible at the new position so the user can
    /// start another drag-based transform immediately.
    fn confirm_transform(state: &mut TransformState, ctx: &mut EventContext) {
        state.cleanup_annotations(ctx);
        state.gizmo.set_highlight(None, ctx);
        state.reset();
        state.sync_gizmo(ctx);
    }

    /// Cancel the transform (restore original state).
    ///
    /// The gizmo remains visible at the original position.
    fn cancel_transform(state: &mut TransformState, ctx: &mut EventContext) {
        state.restore_original_transforms(ctx);
        state.cleanup_annotations(ctx);
        state.gizmo.set_highlight(None, ctx);
        state.reset();
        state.sync_gizmo(ctx);
    }
}

impl Operator for TransformOperator {
    fn activate(&mut self, dispatcher: &mut EventDispatcher) {
        // Keyboard handler for G/R/S activation and X/Y/Z constraints
        let operator_state = self.state.clone();
        let keyboard_callback =
            dispatcher.register(EventKind::KeyboardInput, move |event, ctx| {
                if let Event::KeyboardInput {
                    event: key_event, ..
                } = event
                {
                    // Only handle key press, not release or repeat
                    if key_event.state != ElementState::Pressed || key_event.repeat {
                        return false;
                    }

                    let mut state = operator_state.borrow_mut();

                    match &key_event.logical_key {
                        // Start transform modes (only when inactive)
                        Key::Character('g') | Key::Character('G') if !state.is_active() => {
                            TransformOperator::start_transform(
                                &mut state,
                                TransformMode::Translate,
                                ctx,
                            );
                            state.is_active()
                        }
                        Key::Character('r') | Key::Character('R') if !state.is_active() => {
                            TransformOperator::start_transform(
                                &mut state,
                                TransformMode::Rotate,
                                ctx,
                            );
                            state.is_active()
                        }
                        Key::Character('s') | Key::Character('S') if !state.is_active() => {
                            TransformOperator::start_transform(
                                &mut state,
                                TransformMode::Scale,
                                ctx,
                            );
                            state.is_active()
                        }

                        // Cycle gizmo handles (only when inactive)
                        Key::Character('h') | Key::Character('H') if !state.is_active() => {
                            state.gizmo_mode = match state.gizmo_mode {
                                None => Some(GizmoType::Translate),
                                Some(GizmoType::Translate) => Some(GizmoType::Rotate),
                                Some(GizmoType::Rotate) => Some(GizmoType::Scale),
                                Some(GizmoType::Scale) => None,
                            };
                            state.sync_gizmo(ctx);
                            true
                        }

                        // Axis constraints (only when active)
                        Key::Character('x') | Key::Character('X') if state.is_active() => {
                            state.cycle_axis_constraint('x');
                            state.apply_preview_transform(ctx);
                            state.update_visual_feedback(ctx);
                            let highlight = axis_from_constraint(&state.axis_constraint);
                            state.gizmo.set_highlight(highlight, ctx);
                            true
                        }
                        Key::Character('y') | Key::Character('Y') if state.is_active() => {
                            state.cycle_axis_constraint('y');
                            state.apply_preview_transform(ctx);
                            state.update_visual_feedback(ctx);
                            let highlight = axis_from_constraint(&state.axis_constraint);
                            state.gizmo.set_highlight(highlight, ctx);
                            true
                        }
                        Key::Character('z') | Key::Character('Z') if state.is_active() => {
                            state.cycle_axis_constraint('z');
                            state.apply_preview_transform(ctx);
                            state.update_visual_feedback(ctx);
                            let highlight = axis_from_constraint(&state.axis_constraint);
                            state.gizmo.set_highlight(highlight, ctx);
                            true
                        }

                        // Confirm with Enter (only when active)
                        Key::Named(NamedKey::Enter) if state.is_active() => {
                            TransformOperator::confirm_transform(&mut state, ctx);
                            true
                        }

                        // Cancel with Escape (only when active)
                        Key::Named(NamedKey::Escape) if state.is_active() => {
                            TransformOperator::cancel_transform(&mut state, ctx);
                            true
                        }

                        _ => false,
                    }
                } else {
                    false
                }
            });

        // Mouse motion handler for transform preview
        let operator_state = self.state.clone();
        let motion_callback = dispatcher.register(EventKind::MouseMotion, move |event, ctx| {
            if let Event::MouseMotion { delta } = event {
                let mut state = operator_state.borrow_mut();

                if state.is_active() {
                    // Accumulate mouse delta
                    state.accumulated_delta.0 += delta.0 as f32;
                    state.accumulated_delta.1 += delta.1 as f32;

                    // Apply preview transform
                    state.apply_preview_transform(ctx);
                    true
                } else {
                    false
                }
            } else {
                false
            }
        });

        // Mouse click handler for confirm/cancel
        let operator_state = self.state.clone();
        let click_callback = dispatcher.register(EventKind::MouseClick, move |event, ctx| {
            if let Event::MouseClick { button, .. } = event {
                let mut state = operator_state.borrow_mut();

                if state.is_active() {
                    match button {
                        MouseButton::Left => {
                            // Confirm transform
                            TransformOperator::confirm_transform(&mut state, ctx);
                            true
                        }
                        MouseButton::Right => {
                            // Cancel transform
                            TransformOperator::cancel_transform(&mut state, ctx);
                            true
                        }
                        _ => false,
                    }
                } else {
                    false
                }
            } else {
                false
            }
        });

        // Drag start handler: begin gizmo-based transform when dragging a handle
        let operator_state = self.state.clone();
        let drag_start_callback =
            dispatcher.register(EventKind::MouseDragStart, move |event, ctx| {
                if let Event::MouseDragStart {
                    button: MouseButton::Left,
                    start_pos,
                    ..
                } = event
                {
                    let mut state = operator_state.borrow_mut();

                    // Only handle if no transform is active yet and a gizmo is visible
                    if state.is_active() || !state.gizmo.has_gizmo() {
                        return false;
                    }

                    // Check if drag started on a gizmo handle
                    if let Some(axis) = state.gizmo.pick_handle(start_pos.0, start_pos.1, ctx) {
                        // Determine transform mode from the current gizmo type
                        let mode = match state.gizmo.current_type {
                            Some(GizmoType::Translate) => TransformMode::Translate,
                            Some(GizmoType::Rotate) => TransformMode::Rotate,
                            Some(GizmoType::Scale) => TransformMode::Scale,
                            None => return false,
                        };

                        // Start the transform with the clicked axis constraint
                        TransformOperator::start_transform(&mut state, mode, ctx);
                        if !state.is_active() {
                            return false;
                        }

                        state.axis_constraint = match axis {
                            Axis::X => AxisConstraint::WorldX,
                            Axis::Y => AxisConstraint::WorldY,
                            Axis::Z => AxisConstraint::WorldZ,
                        };
                        state.gizmo.set_highlight(Some(axis), ctx);
                        state.update_visual_feedback(ctx);
                        return true;
                    }
                }
                false
            });

        // Drag handler: update transform while dragging a gizmo handle
        let operator_state = self.state.clone();
        let drag_callback = dispatcher.register(EventKind::MouseDrag, move |event, ctx| {
            if let Event::MouseDrag {
                button: MouseButton::Left,
                delta,
                ..
            } = event
            {
                let mut state = operator_state.borrow_mut();
                if state.is_active() {
                    state.accumulated_delta.0 += delta.0 as f32;
                    state.accumulated_delta.1 += delta.1 as f32;
                    state.apply_preview_transform(ctx);
                    return true;
                }
            }
            false
        });

        // Drag end handler: confirm transform when drag finishes
        let operator_state = self.state.clone();
        let drag_end_callback =
            dispatcher.register(EventKind::MouseDragEnd, move |event, ctx| {
                if let Event::MouseDragEnd {
                    button: MouseButton::Left,
                    ..
                } = event
                {
                    let mut state = operator_state.borrow_mut();
                    if state.is_active() {
                        TransformOperator::confirm_transform(&mut state, ctx);
                        return true;
                    }
                }
                false
            });

        // Cursor move handler: hover highlight on gizmo handles
        let operator_state = self.state.clone();
        let cursor_callback =
            dispatcher.register(EventKind::CursorMoved, move |event, ctx| {
                if let Event::CursorMoved { position } = event {
                    let mut state = operator_state.borrow_mut();

                    // Only do hover highlighting when gizmo is visible but no transform active
                    if state.gizmo.has_gizmo() && !state.is_active() {
                        let axis = state.gizmo.pick_handle(position.0 as f32, position.1 as f32, ctx);
                        state.gizmo.set_highlight(axis, ctx);
                    }
                }
                false // Never consume cursor move events
            });

        // Update handler: keep gizmo in sync with selection changes
        let operator_state = self.state.clone();
        let update_callback = dispatcher.register(EventKind::Update, move |_event, ctx| {
            let mut state = operator_state.borrow_mut();
            if state.gizmo_mode.is_some() && !state.is_active() {
                state.sync_gizmo(ctx);
            }
            false // Never consume update events
        });

        self.callback_ids = vec![
            keyboard_callback,
            motion_callback,
            click_callback,
            drag_start_callback,
            drag_callback,
            drag_end_callback,
            cursor_callback,
            update_callback,
        ];
    }

    fn deactivate(&mut self, dispatcher: &mut EventDispatcher) {
        for id in &self.callback_ids {
            dispatcher.unregister(*id);
        }
        self.callback_ids.clear();
    }

    fn id(&self) -> OperatorId {
        self.id
    }

    fn name(&self) -> &str {
        "Transform"
    }

    fn callback_ids(&self) -> &[CallbackId] {
        &self.callback_ids
    }

    fn is_active(&self) -> bool {
        !self.callback_ids.is_empty()
    }
}
