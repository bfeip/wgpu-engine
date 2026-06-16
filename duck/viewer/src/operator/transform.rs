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

use std::collections::HashMap;

use duck_engine_common::{
    EuclideanSpace, InnerSpace, Matrix4, Point3, Quaternion, Rotation, SquareMatrix, Vector3,
};
use duck_engine_scene::{Mesh, NodeFlags, Scene, common};

use crate::common::{
    apply_scale, centroid_of_slice, compose_rotation, decompose_matrix, local_axis_x, local_axis_y,
    local_axis_z, quaternion_from_axis_angle_safe, rotate_position_about_pivot,
    scale_position_about_pivot_local, scale_position_about_pivot_world, RgbaColor, Transform,
};
use crate::common::Axis;
use serde::{Deserialize, Serialize};

use crate::bindings::{InputBinding, InputMap};
use crate::event::{Event, EventContext};
use crate::geom_query::{pick_all_from_ray, RayPickQuery};
use crate::input::{ElementState, Key, Modifiers, MouseButton, NamedKey};
use crate::operator::Operator;
use crate::gizmo::{self, GizmoType};
use crate::scene::{
    DisplayBehavior, FaceMaterialId, Instance, LineMaterial, LineMaterialId, MeshId, NodeId,
    RenderLayer,
};
use crate::scene_scale;

/// Semantic actions for the transform operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransformAction {
    /// Begin a grab (translate) operation.
    StartTranslate,
    /// Begin a rotate operation.
    StartRotate,
    /// Begin a scale operation.
    StartScale,
    /// Cycle the persistent gizmo handle (None → Translate → Rotate → Scale → None).
    CycleGizmo,
    /// Cycle the axis constraint to X (world then local on repeated press).
    ConstrainX,
    /// Cycle the axis constraint to Y (world then local on repeated press).
    ConstrainY,
    /// Cycle the axis constraint to Z (world then local on repeated press).
    ConstrainZ,
    /// Confirm the active transform via keyboard.
    KeyConfirm,
    /// Cancel the active transform via keyboard.
    KeyCancel,
    /// Confirm the active transform via mouse click.
    MouseConfirm,
    /// Cancel the active transform via mouse click.
    MouseCancel,
    /// Drag interaction with a gizmo handle (drag start, drag, and drag end).
    GizmoDrag,
}

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
    /// Root node for all gizmo geometry. Created when first needed.
    root_node: Option<NodeId>,
    /// Node IDs of the gizmo handles (one per axis: X, Y, Z).
    node_ids: Vec<NodeId>,
    /// Mesh IDs added to the scene for gizmo geometry.
    mesh_ids: Vec<MeshId>,
    /// Material IDs added to the scene for gizmo handles.
    material_ids: Vec<FaceMaterialId>,
    /// Which axis is currently highlighted (hovered or active).
    highlighted_axis: Option<Axis>,
    /// Current gizmo type being displayed.
    current_type: Option<GizmoType>,
}

impl GizmoState {
    fn new() -> Self {
        Self {
            root_node: None,
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
    fn show(&mut self, gizmo_type: GizmoType, pivot: Point3, size: f32, ctx: &mut EventContext) {
        let mut scene = ctx.scene.lock().unwrap();

        self.hide(&mut scene);

        self.root_node.get_or_insert_with(|| {
            let id = scene.add_node(
                None, Some("Gizmo root".to_owned()), Transform::IDENTITY, NodeFlags::DO_NOT_EXPORT
            ).expect("Failed to create Gizmo root node");
            // Draw the gizmo on the overlay layer; handles inherit it.
            scene.set_node_display(id, DisplayBehavior { layer: RenderLayer::Overlay, ..Default::default() });
            id
        });

        let handles = gizmo::build_handles(gizmo_type, size);
        let pivot_transform = common::Transform::from_position(pivot);

        for handle in handles {
            let mesh_id = scene.add_mesh(handle.mesh);
            let material_id = scene.add_face_material(handle.material);
            let node_id = scene
                .add_instance_node(
                    self.root_node,
                    Instance::new(mesh_id).with_face_material(material_id),
                    None,
                    pivot_transform,
                    NodeFlags::DO_NOT_EXPORT
                )
                .expect("Failed to add gizmo node");

            self.node_ids.push(node_id);
            self.mesh_ids.push(mesh_id);
            self.material_ids.push(material_id);
        }

        self.current_type = Some(gizmo_type);
    }

    /// Remove all gizmo geometry from the scene.
    fn hide(&mut self, scene: &mut Scene) {
        for &node_id in &self.node_ids {
            scene.remove_node(node_id);
        }
        for &mesh_id in &self.mesh_ids {
            scene.remove_mesh(mesh_id);
        }
        for &material_id in &self.material_ids {
            scene.remove_face_material(material_id);
        }

        self.node_ids.clear();
        self.mesh_ids.clear();
        self.material_ids.clear();
        self.highlighted_axis = None;
        self.current_type = None;
    }

    /// Update the gizmo position (e.g. when pivot changes).
    fn update_position(&self, pivot: Point3, ctx: &mut EventContext) {
        let mut scene = ctx.scene.lock().unwrap();
        for &node_id in &self.node_ids {
            if scene.has_node(node_id) {
                scene.set_node_position(node_id, pivot);
            }
        }
    }

    /// Pick which gizmo handle (if any) is under the given screen position.
    fn pick_handle(&self, cursor_x: f32, cursor_y: f32, ctx: &EventContext) -> Option<Axis> {
        if !self.has_gizmo() {
            return None;
        }

        let ray = ctx.camera().ray_from_screen_point(cursor_x, cursor_y, ctx.size.0, ctx.size.1);
        let scene = ctx.scene.lock().unwrap();
        let results = pick_all_from_ray(&RayPickQuery::faces(ray), &*scene);

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

        let mut scene = ctx.scene.lock().unwrap();

        // Restore previous highlight to normal color
        if let Some(prev_axis) = self.highlighted_axis {
            let idx = axis_index(prev_axis);
            if let Some(&mat_id) = self.material_ids.get(idx)
                && let Some(mat) = scene.get_face_material_mut(mat_id) {
                    mat.set_base_color_factor(prev_axis.color());
                }
        }

        // Apply highlight color to new axis
        if let Some(new_axis) = axis {
            let idx = axis_index(new_axis);
            if let Some(&mat_id) = self.material_ids.get(idx)
                && let Some(mat) = scene.get_face_material_mut(mat_id) {
                    mat.set_base_color_factor(gizmo::highlight_color(new_axis));
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

/// Operator for Blender-style transform operations (grab/rotate/scale).
pub struct TransformOperator {
    /// Current transform mode (None when inactive).
    mode: Option<TransformMode>,

    /// Current axis constraint.
    axis_constraint: AxisConstraint,

    /// Original transforms of selected nodes (for cancel/restore).
    original_transforms: Vec<OriginalTransform>,

    /// The rotation of the primary selected node (for local axis transforms).
    primary_rotation: Quaternion,

    /// Accumulated mouse movement since transform started.
    accumulated_delta: (f32, f32),

    /// Center point of selection in world space (pivot point for rotation/scale).
    pivot_world: Point3,

    /// Model radius for scaling sensitivity.
    model_radius: f32,

    /// Gizmo handle state (3D visual handles for the active transform).
    gizmo: GizmoState,

    /// Which gizmo type to display (persists across transforms, cycled by H key).
    /// Independent of `mode` — the gizmo is a persistent visual tool, while
    /// `mode` tracks the active keyboard-driven transform.
    gizmo_mode: Option<GizmoType>,

    /// Root node for transform annotations, created when needed.
    annotation_root: Option<NodeId>,

    /// Transform annotations, cleaned up after transform is complete
    annotations: Vec<NodeId>,

    /// Materials for the colored annotations (So we're not making dozens of copies)
    annotation_axis_materials: HashMap<Axis, LineMaterialId>,

    /// Nodes whose transform was just confirmed, awaiting collection via
    /// [`Self::take_committed`].
    ///
    /// NOTE: this exists solely so the modeler can bake a committed
    /// transform into its underlying CAD geometry — the viewer itself has no use
    /// for it. It is a quick-fix; if this operator ever grows
    /// a proper commit-callback or event mechanism, remove this in favor of that.
    committed: Option<Vec<NodeId>>,

    pub bindings: InputMap<TransformAction>,
}

impl TransformOperator {
    /// Creates a new transform operator with default bindings.
    pub fn new() -> Self {
        let bindings = InputMap::new()
            .bind(
                InputBinding::Key { key: Key::Character('g'), modifiers: Modifiers::default() },
                TransformAction::StartTranslate,
            )
            .bind(
                InputBinding::Key { key: Key::Character('r'), modifiers: Modifiers::default() },
                TransformAction::StartRotate,
            )
            .bind(
                InputBinding::Key { key: Key::Character('s'), modifiers: Modifiers::default() },
                TransformAction::StartScale,
            )
            .bind(
                InputBinding::Key { key: Key::Character('h'), modifiers: Modifiers::default() },
                TransformAction::CycleGizmo,
            )
            .bind(
                InputBinding::Key { key: Key::Character('x'), modifiers: Modifiers::default() },
                TransformAction::ConstrainX,
            )
            .bind(
                InputBinding::Key { key: Key::Character('y'), modifiers: Modifiers::default() },
                TransformAction::ConstrainY,
            )
            .bind(
                InputBinding::Key { key: Key::Character('z'), modifiers: Modifiers::default() },
                TransformAction::ConstrainZ,
            )
            .bind(
                InputBinding::Key { key: Key::Named(NamedKey::Enter), modifiers: Modifiers::default() },
                TransformAction::KeyConfirm,
            )
            .bind(
                InputBinding::Key { key: Key::Named(NamedKey::Escape), modifiers: Modifiers::default() },
                TransformAction::KeyCancel,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
                TransformAction::MouseConfirm,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Right, modifiers: Modifiers::default() },
                TransformAction::MouseCancel,
            )
            .bind(
                InputBinding::MouseDragStart { button: MouseButton::Left, modifiers: Modifiers::default() },
                TransformAction::GizmoDrag,
            )
            .bind(
                InputBinding::MouseDrag { button: MouseButton::Left, modifiers: Modifiers::default() },
                TransformAction::GizmoDrag,
            )
            .bind(
                InputBinding::MouseDragEnd { button: MouseButton::Left, modifiers: Modifiers::default() },
                TransformAction::GizmoDrag,
            );
        Self {
            mode: None,
            axis_constraint: AxisConstraint::None,
            original_transforms: Vec::new(),
            primary_rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            accumulated_delta: (0.0, 0.0),
            pivot_world: Point3::origin(),
            model_radius: 1.0,
            gizmo: GizmoState::new(),
            gizmo_mode: None,
            annotation_root: None,
            annotations: Vec::new(),
            annotation_axis_materials: HashMap::new(),
            committed: None,
            bindings,
        }
    }

    /// Returns true if a transform operation is currently active.
    pub fn is_active(&self) -> bool {
        self.mode.is_some()
    }

    /// Takes the set of nodes whose transform was confirmed since the last call,
    /// clearing the internal record. Returns `None` if nothing was confirmed.
    ///
    /// NOTE: for the modeler — see the `committed` field. Cancelled
    /// transforms are *not* reported here (the operator restores the original
    /// transforms on cancel).
    pub fn take_committed(&mut self) -> Option<Vec<NodeId>> {
        self.committed.take()
    }

    /// Tears down all scene-side visuals owned by this operator (gizmo handles
    /// and annotation lines) and aborts any in-progress transform, restoring the
    /// affected nodes to their pre-transform state.
    pub fn teardown(&mut self, scene: &mut Scene) {
        if self.is_active() {
            for original in &self.original_transforms {
                scene.set_node_transform(original.node_id, original.local_transform);
            }
        }
        for id in self.annotations.drain(..) {
            scene.remove_node(id);
        }
        self.gizmo.hide(scene);
        self.gizmo_mode = None;
        self.reset();
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
    fn get_constraint_axis(&self) -> Option<Vector3> {
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
    fn compute_translation(&self, ctx: &EventContext) -> Vector3 {
        let camera = ctx.camera();
        let pivot = &self.pivot_world;
        let (width, height) = ctx.size;
        let (dx, dy) = self.accumulated_delta;

        let movement_plane = common::Plane::from_point(camera.forward(), *pivot);
        let Point3 { x: screen_x, y: screen_y, .. } = camera.project_point_screen(*pivot, width, height);
        let diff_ray = camera.ray_from_screen_point(screen_x + dx, screen_y + dy, width, height);
        let new_pivot = diff_ray.intersect_plane(&movement_plane)
            .map_or(*pivot, |intersection| intersection.1);
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
    fn compute_rotation(&self, ctx: &EventContext) -> Quaternion {
        // 0.5 degrees per pixel
        let sensitivity = 0.5_f32.to_radians();
        let angle = self.accumulated_delta.0 * sensitivity;

        let axis = match self.get_constraint_axis() {
            None => {
                // Free rotation: rotate around view axis
                ctx.camera().forward()
            }
            Some(axis) => axis,
        };

        quaternion_from_axis_angle_safe(axis, angle)
    }

    /// Compute the scale factor based on mouse movement and constraints.
    fn compute_scale(&self) -> Vector3 {
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

        // Pre-compute camera-dependent values before locking the scene.
        // ctx.camera() acquires and releases the scene lock internally.
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

        // Now apply transforms to nodes under a single scene lock.
        let mut scene = ctx.scene.lock().unwrap();
        for orig in &self.original_transforms {
            let inv_parent = orig
                .parent_world_transform
                .to_matrix()
                .invert()
                .unwrap_or(Matrix4::identity());

            if !scene.has_node(orig.node_id) {
                continue;
            }

            match mode {
                TransformMode::Translate => {
                    let delta = translation_delta.unwrap();
                    let new_world_pos = orig.world_transform.position + delta;
                    let new_local_pos =
                        Point3::from_homogeneous(inv_parent * new_world_pos.to_homogeneous());
                    scene.set_node_position(orig.node_id, new_local_pos);
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
                    scene.set_node_position(orig.node_id, new_local_pos);

                    // Convert world rotation to local space
                    let pr = orig.parent_world_transform.rotation;
                    let pr_inv = pr.conjugate();
                    let local_rotation = pr_inv * rotation * pr;
                    let new_rotation =
                        compose_rotation(orig.local_transform.rotation, local_rotation);
                    scene.set_node_rotation(orig.node_id, new_rotation);
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
                        scene.set_node_position(orig.node_id, new_local_pos);
                        let new_scale = apply_scale(orig.local_transform.scale, scale);
                        scene.set_node_scale(orig.node_id, new_scale);
                    } else {
                        // World axis: scale world position around pivot, convert to local
                        let new_world_pos = scale_position_about_pivot_world(
                            orig.world_transform.position,
                            self.pivot_world,
                            scale,
                        );
                        let new_local_pos =
                            Point3::from_homogeneous(inv_parent * new_world_pos.to_homogeneous());
                        scene.set_node_position(orig.node_id, new_local_pos);

                        // Convert world-axis scale to local space
                        let pr_inv = orig.parent_world_transform.rotation.conjugate();
                        let local_scale = world_scale_to_local(scale, pr_inv);
                        let new_scale = apply_scale(orig.local_transform.scale, local_scale);
                        scene.set_node_scale(orig.node_id, new_scale);
                    }
                }
            }
        }
    }

    /// Restore all nodes to their original transforms.
    fn restore_original_transforms(&self, ctx: &mut EventContext) {
        let mut scene = ctx.scene.lock().unwrap();
        for orig in &self.original_transforms {
            if scene.has_node(orig.node_id) {
                scene.set_node_transform(orig.node_id, orig.local_transform);
            }
        }
    }

    /// Update visual feedback annotations.
    fn update_visual_feedback(&mut self, ctx: &mut EventContext) {
        let mut scene = ctx.scene.lock().unwrap();

        // Create annotation root node if it does not exist
        self.annotation_root.get_or_insert(
            scene.add_node(
                None, Some("Transform annotations".to_owned()), Transform::IDENTITY, NodeFlags::inert()
            ).expect("Failed to create transform annotation root node")
        );

        // Clear previous annotations
        for id in self.annotations.drain(..) {
            scene.remove_node(id);
        }

        // Add axis constraint line if constrained
        if let Some(color) = self.axis_constraint.color()
            && let Some(axis) = self.get_constraint_axis() {
                let half_length = self.model_radius * 2.0;
                let start = self.pivot_world - axis * half_length;
                let end = self.pivot_world + axis * half_length;
                let mesh = Mesh::line(start, end);
                let mesh_id = scene.add_mesh(mesh);

                // Get or insert the material for this axis annotation
                let create_color_material = |scene: &mut Scene| {
                    scene.add_line_material(LineMaterial::new(color))
                };
                let mut material = self.annotation_axis_materials.entry(
                    axis_from_constraint(&self.axis_constraint).unwrap()
                ).or_insert(create_color_material(&mut *scene)).to_owned();
                if scene.get_line_material(material).is_none() {
                    // Our material was removed from the scene since we last used it.
                    // This can happen if, while unused, the scene removed all unreferenced
                    // resources. We'll have to reinsert the material.
                    material = create_color_material(&mut *scene);
                }

                let id = scene.add_instance_node(
                    self.annotation_root,
                    Instance::new(mesh_id).with_line_material(material),
                    Some("Transform axis annotation".to_owned()),
                    Transform::IDENTITY,
                    NodeFlags::inert()
                ).expect("Failed to create axis annotation");
                self.annotations.push(id);
            }
    }

    /// Clean up annotations.
    fn cleanup_annotations(&mut self, ctx: &mut EventContext) {
        let mut scene = ctx.scene.lock().unwrap();
        for id in self.annotations.drain(..) {
            scene.remove_node(id);
        }
    }

    /// Show, reposition, or hide the gizmo based on `gizmo_mode` and selection.
    fn sync_gizmo(&mut self, ctx: &mut EventContext) {
        let selected = ctx.selection.selected_nodes();
        match (self.gizmo_mode, selected.is_empty()) {
            (Some(gizmo_type), false) => {
                let (positions, model_radius) = {
                    let scene = ctx.scene.lock().unwrap();
                    let positions: Vec<Point3> = selected
                        .iter()
                        .filter_map(|&nid| {
                            scene.nodes_bounding(nid).bounds.map(|aabb| aabb.center())
                        })
                        .collect();
                    let model_radius =
                        scene_scale::model_radius_from_bounds(scene.bounding().bounds.as_ref());
                    (positions, model_radius)
                };
                let pivot = centroid_of_slice(&positions).unwrap_or(Point3::origin());

                if self.gizmo.current_type == Some(gizmo_type) {
                    self.gizmo.update_position(pivot, ctx);
                } else {
                    self.gizmo.show(gizmo_type, pivot, model_radius * 0.15, ctx);
                }
            }
            _ => {
                let mut scene = ctx.scene.lock().unwrap();
                self.gizmo.hide(&mut scene);
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

    /// Start a transform operation with the given mode.
    fn start_transform(&mut self, mode: TransformMode, ctx: &mut EventContext) {
        // Get selected nodes
        let selected_nodes = ctx.selection.selected_nodes();
        if selected_nodes.is_empty() {
            return;
        }

        // Store original transforms with world-space info
        let mut world_positions: Vec<Point3> = Vec::new();
        {
            let scene = ctx.scene.lock().unwrap();
            for node_id in &selected_nodes {
                if let Some(node) = scene.get_node(*node_id) {
                    let Some(world_matrix) = scene.nodes_transform(*node_id) else { continue };
                    let world_transform = decompose_matrix(&world_matrix);

                    let parent_world_transform = if let Some(parent_id) = node.parent() {
                        scene.nodes_transform(parent_id)
                            .map(|m| decompose_matrix(&m))
                            .unwrap_or(Transform::IDENTITY)
                    } else {
                        Transform::IDENTITY
                    };

                    let world_pos = scene.nodes_bounding(*node_id).bounds
                        .map(|aabb| aabb.center())
                        .unwrap_or(world_transform.position);
                    world_positions.push(world_pos);
                    self.original_transforms.push(OriginalTransform {
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

            // Store primary selection's rotation for local axis transforms
            if let Some(primary) = ctx.selection.primary()
                && let Some(node) = scene.get_node(primary.node_id()) {
                    self.primary_rotation = node.rotation();
                }

            // Get model radius for sensitivity scaling
            self.model_radius =
                scene_scale::model_radius_from_bounds(scene.bounding().bounds.as_ref());
        }

        // Compute pivot as centroid of world-space positions
        self.pivot_world = centroid_of_slice(&world_positions).unwrap_or(Point3::origin());

        // Activate the transform
        self.mode = Some(mode);
        self.axis_constraint = AxisConstraint::None;
        self.accumulated_delta = (0.0, 0.0);
    }

    /// Confirm the transform (keep current state).
    ///
    /// The gizmo remains visible at the new position so the user can
    /// start another drag-based transform immediately.
    fn confirm_transform(&mut self, ctx: &mut EventContext) {
        // Record the confirmed nodes for downstream consumers (modeler CAD sync)
        // before `reset()` clears `original_transforms`.
        self.committed = Some(self.original_transforms.iter().map(|o| o.node_id).collect());
        self.cleanup_annotations(ctx);
        self.gizmo.set_highlight(None, ctx);
        self.reset();
        self.sync_gizmo(ctx);
    }

    /// Cancel the transform (restore original state).
    ///
    /// The gizmo remains visible at the original position.
    fn cancel_transform(&mut self, ctx: &mut EventContext) {
        self.restore_original_transforms(ctx);
        self.cleanup_annotations(ctx);
        self.gizmo.set_highlight(None, ctx);
        self.reset();
        self.sync_gizmo(ctx);
    }
}

/// Converts a world-axis-constrained scale into the parent's local space.
///
/// For uniform scale (no constraint), returns as-is. For single-axis world
/// constraints, rotates the scale axis into local space and decomposes it
/// into per-axis scale contributions.
fn world_scale_to_local(scale: Vector3, parent_rotation_inv: Quaternion) -> Vector3 {
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

impl Operator for TransformOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        match event {
            Event::KeyboardInput { event: key_event, .. } => {
                if key_event.state != ElementState::Pressed || key_event.repeat {
                    return false;
                }
                let actions = self.bindings
                    .actions_for_key(&key_event.logical_key, ctx.modifiers)
                    .to_vec();
                for action in actions {
                    match action {
                        TransformAction::StartTranslate if !self.is_active() => {
                            self.start_transform(TransformMode::Translate, ctx);
                            return self.is_active();
                        }
                        TransformAction::StartRotate if !self.is_active() => {
                            self.start_transform(TransformMode::Rotate, ctx);
                            return self.is_active();
                        }
                        TransformAction::StartScale if !self.is_active() => {
                            self.start_transform(TransformMode::Scale, ctx);
                            return self.is_active();
                        }
                        TransformAction::CycleGizmo if !self.is_active() => {
                            self.gizmo_mode = match self.gizmo_mode {
                                None => Some(GizmoType::Translate),
                                Some(GizmoType::Translate) => Some(GizmoType::Rotate),
                                Some(GizmoType::Rotate) => Some(GizmoType::Scale),
                                Some(GizmoType::Scale) => None,
                            };
                            self.sync_gizmo(ctx);
                            return true;
                        }
                        TransformAction::ConstrainX if self.is_active() => {
                            self.cycle_axis_constraint('x');
                            self.apply_preview_transform(ctx);
                            self.update_visual_feedback(ctx);
                            let highlight = axis_from_constraint(&self.axis_constraint);
                            self.gizmo.set_highlight(highlight, ctx);
                            return true;
                        }
                        TransformAction::ConstrainY if self.is_active() => {
                            self.cycle_axis_constraint('y');
                            self.apply_preview_transform(ctx);
                            self.update_visual_feedback(ctx);
                            let highlight = axis_from_constraint(&self.axis_constraint);
                            self.gizmo.set_highlight(highlight, ctx);
                            return true;
                        }
                        TransformAction::ConstrainZ if self.is_active() => {
                            self.cycle_axis_constraint('z');
                            self.apply_preview_transform(ctx);
                            self.update_visual_feedback(ctx);
                            let highlight = axis_from_constraint(&self.axis_constraint);
                            self.gizmo.set_highlight(highlight, ctx);
                            return true;
                        }
                        TransformAction::KeyConfirm if self.is_active() => {
                            self.confirm_transform(ctx);
                            return true;
                        }
                        TransformAction::KeyCancel if self.is_active() => {
                            self.cancel_transform(ctx);
                            return true;
                        }
                        _ => {}
                    }
                }
                false
            }

            Event::MouseMotion { delta } => {
                if self.is_active() {
                    self.accumulated_delta.0 += delta.0 as f32;
                    self.accumulated_delta.1 += delta.1 as f32;
                    self.apply_preview_transform(ctx);
                    true
                } else {
                    false
                }
            }

            Event::MouseClick { button, .. } => {
                if !self.is_active() {
                    return false;
                }
                let actions = self.bindings.actions_for_click(*button, ctx.modifiers).to_vec();
                for action in actions {
                    match action {
                        TransformAction::MouseConfirm => {
                            self.confirm_transform(ctx);
                            return true;
                        }
                        TransformAction::MouseCancel => {
                            self.cancel_transform(ctx);
                            return true;
                        }
                        _ => {}
                    }
                }
                false
            }

            Event::MouseDragStart { button, start_pos, .. } => {
                if !self.bindings
                    .actions_for_drag_start(*button, ctx.modifiers)
                    .contains(&TransformAction::GizmoDrag)
                {
                    return false;
                }
                if self.is_active() || !self.gizmo.has_gizmo() {
                    return false;
                }
                if let Some(axis) = self.gizmo.pick_handle(start_pos.0, start_pos.1, ctx) {
                    let mode = match self.gizmo.current_type {
                        Some(GizmoType::Translate) => TransformMode::Translate,
                        Some(GizmoType::Rotate) => TransformMode::Rotate,
                        Some(GizmoType::Scale) => TransformMode::Scale,
                        None => return false,
                    };
                    self.start_transform(mode, ctx);
                    if !self.is_active() {
                        return false;
                    }
                    self.axis_constraint = match axis {
                        Axis::X => AxisConstraint::WorldX,
                        Axis::Y => AxisConstraint::WorldY,
                        Axis::Z => AxisConstraint::WorldZ,
                    };
                    self.gizmo.set_highlight(Some(axis), ctx);
                    self.update_visual_feedback(ctx);
                    return true;
                }
                false
            }

            Event::MouseDrag { button, delta, .. } => {
                if !self.bindings
                    .actions_for_drag(*button, ctx.modifiers)
                    .contains(&TransformAction::GizmoDrag)
                {
                    return false;
                }
                if self.is_active() {
                    self.accumulated_delta.0 += delta.0;
                    self.accumulated_delta.1 += delta.1;
                    self.apply_preview_transform(ctx);
                    return true;
                }
                false
            }

            Event::MouseDragEnd { button, .. } => {
                if !self.bindings
                    .actions_for_drag_end(*button, Modifiers::default())
                    .contains(&TransformAction::GizmoDrag)
                {
                    return false;
                }
                if self.is_active() {
                    self.confirm_transform(ctx);
                    return true;
                }
                false
            }

            Event::CursorMoved { position } => {
                // Hover highlight on gizmo handles when gizmo is visible but no transform active
                if self.gizmo.has_gizmo() && !self.is_active() {
                    let axis = self.gizmo.pick_handle(position.0 as f32, position.1 as f32, ctx);
                    self.gizmo.set_highlight(axis, ctx);
                }
                false
            }

            Event::Update { .. } => {
                if self.gizmo_mode.is_some() && !self.is_active() {
                    self.sync_gizmo(ctx);
                }
                false
            }

            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Transform"
    }
}
