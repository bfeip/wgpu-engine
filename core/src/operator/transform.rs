//! Transform operator for Blender-style grab/rotate/scale operations.
//!
//! Controls:
//! - G: Start grab/translate mode
//! - R: Start rotate mode
//! - S: Start scale mode
//! - X/Y/Z: Constrain to axis (toggle: none → world → local → none)
//! - Left-click / Enter: Confirm transform
//! - Right-click / Escape: Cancel transform
//! - Mouse movement: Adjust transform magnitude

use std::cell::RefCell;
use std::rc::Rc;

use cgmath::{EuclideanSpace, InnerSpace, Point3, Quaternion, Rotation, Rotation3, Vector3};

use crate::common::RgbaColor;
use crate::event::{CallbackId, Event, EventContext, EventDispatcher, EventKind};
use crate::input::{ElementState, Key, MouseButton, NamedKey};
use crate::operator::{Operator, OperatorId};
use crate::scene::annotation::AnnotationId;
use crate::scene::NodeId;
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

/// Original transform state for a node (used for cancel/restore).
#[derive(Debug, Clone)]
struct OriginalTransform {
    node_id: NodeId,
    position: Point3<f32>,
    rotation: Quaternion<f32>,
    scale: Vector3<f32>,
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
            AxisConstraint::LocalX => Some(self.primary_rotation.rotate_vector(Vector3::unit_x())),
            AxisConstraint::LocalY => Some(self.primary_rotation.rotate_vector(Vector3::unit_y())),
            AxisConstraint::LocalZ => Some(self.primary_rotation.rotate_vector(Vector3::unit_z())),
        }
    }

    /// Compute the translation delta based on mouse movement and constraints.
    fn compute_translation(&self, ctx: &EventContext) -> Vector3<f32> {
        let sensitivity = self.model_radius * 0.002;
        let (dx, dy) = self.accumulated_delta;

        match self.get_constraint_axis() {
            None => {
                // Free translation: move in camera plane
                let camera = ctx.renderer.camera();
                let right = camera.right();
                let up = camera.up;
                right * dx * sensitivity + up * (-dy) * sensitivity
            }
            Some(axis) => {
                // Constrained: project mouse movement onto axis
                // Use horizontal mouse movement as primary input
                axis * dx * sensitivity
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
                ctx.renderer.camera().forward()
            }
            Some(axis) => axis,
        };

        if axis.magnitude2() > 0.0001 {
            Quaternion::from_axis_angle(axis.normalize(), cgmath::Rad(angle))
        } else {
            Quaternion::new(1.0, 0.0, 0.0, 0.0)
        }
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
            let node = match ctx.scene.get_node_mut(orig.node_id) {
                Some(n) => n,
                None => continue,
            };

            match mode {
                TransformMode::Translate => {
                    let delta = translation_delta.unwrap();
                    node.set_position(orig.position + delta);
                }
                TransformMode::Rotate => {
                    let rotation = rotation_quat.unwrap();
                    // Rotate position around pivot
                    let offset = orig.position - self.pivot_world;
                    let rotated_offset = rotation.rotate_vector(offset);
                    node.set_position(self.pivot_world + rotated_offset);
                    // Also rotate the node's orientation
                    node.set_rotation(rotation * orig.rotation);
                }
                TransformMode::Scale => {
                    let scale = scale_factor.unwrap();

                    // For local axis constraints, we need to scale in local space
                    if self.axis_constraint.is_local() {
                        // Scale in local space: transform scale vector by node's rotation
                        let local_scale = Vector3::new(
                            orig.scale.x * scale.x,
                            orig.scale.y * scale.y,
                            orig.scale.z * scale.z,
                        );
                        node.set_scale(local_scale);

                        // Scale position offset from pivot in local space
                        let offset = orig.position - self.pivot_world;
                        let local_offset = self
                            .primary_rotation
                            .conjugate()
                            .rotate_vector(offset);
                        let scaled_local = Vector3::new(
                            local_offset.x * scale.x,
                            local_offset.y * scale.y,
                            local_offset.z * scale.z,
                        );
                        let world_offset = self.primary_rotation.rotate_vector(scaled_local);
                        node.set_position(self.pivot_world + world_offset);
                    } else {
                        // Scale in world space
                        let offset = orig.position - self.pivot_world;
                        let scaled_offset = Vector3::new(
                            offset.x * scale.x,
                            offset.y * scale.y,
                            offset.z * scale.z,
                        );
                        node.set_position(self.pivot_world + scaled_offset);
                        node.set_scale(Vector3::new(
                            orig.scale.x * scale.x,
                            orig.scale.y * scale.y,
                            orig.scale.z * scale.z,
                        ));
                    }
                }
            }
        }
    }

    /// Restore all nodes to their original transforms.
    fn restore_original_transforms(&self, ctx: &mut EventContext) {
        for orig in &self.original_transforms {
            if let Some(node) = ctx.scene.get_node_mut(orig.node_id) {
                node.set_position(orig.position);
                node.set_rotation(orig.rotation);
                node.set_scale(orig.scale);
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

    /// Reset state after transform completes (confirm or cancel).
    fn reset(&mut self) {
        self.mode = None;
        self.axis_constraint = AxisConstraint::None;
        self.original_transforms.clear();
        self.accumulated_delta = (0.0, 0.0);
    }
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

        // Store original transforms and compute pivot
        let mut pivot_sum = Vector3::new(0.0, 0.0, 0.0);
        let mut count = 0;

        for node_id in &selected_nodes {
            if let Some(node) = ctx.scene.get_node(*node_id) {
                state.original_transforms.push(OriginalTransform {
                    node_id: *node_id,
                    position: node.position(),
                    rotation: node.rotation(),
                    scale: node.scale(),
                });
                pivot_sum += node.position().to_vec();
                count += 1;
            }
        }

        if count == 0 {
            return;
        }

        // Compute pivot as centroid of selected nodes
        state.pivot_world = Point3::from_vec(pivot_sum / count as f32);

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
    fn confirm_transform(state: &mut TransformState, ctx: &mut EventContext) {
        state.cleanup_annotations(ctx);
        state.reset();
    }

    /// Cancel the transform (restore original state).
    fn cancel_transform(state: &mut TransformState, ctx: &mut EventContext) {
        state.restore_original_transforms(ctx);
        state.cleanup_annotations(ctx);
        state.reset();
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

                        // Axis constraints (only when active)
                        Key::Character('x') | Key::Character('X') if state.is_active() => {
                            state.cycle_axis_constraint('x');
                            state.apply_preview_transform(ctx);
                            state.update_visual_feedback(ctx);
                            true
                        }
                        Key::Character('y') | Key::Character('Y') if state.is_active() => {
                            state.cycle_axis_constraint('y');
                            state.apply_preview_transform(ctx);
                            state.update_visual_feedback(ctx);
                            true
                        }
                        Key::Character('z') | Key::Character('Z') if state.is_active() => {
                            state.cycle_axis_constraint('z');
                            state.apply_preview_transform(ctx);
                            state.update_visual_feedback(ctx);
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

        self.callback_ids = vec![keyboard_callback, motion_callback, click_callback];
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
