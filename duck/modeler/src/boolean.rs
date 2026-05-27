use anyhow::{Context, Result};
use duck_engine_common::RgbaColor;
use duck_engine_scene::NodeId;
use duck_engine_scene::cad::{CadTessellationOptions, tessellate_into};
use duck_engine_viewer::scene::Scene;
use opencascade::primitives::Shape;

use crate::document::{CadDocument, PartId};
use crate::part_map::PartNodeMap;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum BooleanKind {
    #[default]
    Subtract,
    Union,
    Intersect,
}

struct BooleanResult {
    shape: Shape,
    color: RgbaColor,
    target_part_id: PartId,
    tool_part_ids: Vec<PartId>,
}

/// Resolve nodes to shapes and compute the boolean result, without touching the scene.
fn compute_boolean(
    kind: BooleanKind,
    target: NodeId,
    tools: &[NodeId],
    document: &CadDocument,
    part_map: &PartNodeMap,
) -> Result<BooleanResult> {
    let target_part_id = part_map.part_for_node(target)
        .context("Target node is not a known CAD part")?;
    let tool_part_ids: Vec<_> = tools.iter()
        .map(|&node| part_map.part_for_node(node).context("Tool node is not a known CAD part"))
        .collect::<Result<_>>()?;

    let target_part = document.get_part(target_part_id)
        .context("Target part not found")?;
    let color = target_part.color;
    let tool_shapes: Vec<_> = tool_part_ids.iter()
        .map(|&id| document.get_part(id).map(|p| p.shape.clone()).context("Tool part not found"))
        .collect::<Result<_>>()?;

    let mut shape = target_part.shape.clone();
    for tool in &tool_shapes {
        shape = match kind {
            BooleanKind::Subtract  => shape.subtract(tool).shape,
            BooleanKind::Union     => shape.union(tool).shape,
            BooleanKind::Intersect => shape.intersect(tool).shape,
        };
    }

    Ok(BooleanResult { shape, color, target_part_id, tool_part_ids })
}

pub fn execute_boolean(
    kind: BooleanKind,
    target: NodeId,
    tools: &[NodeId],
    scene: &mut Scene,
    document: &mut CadDocument,
    part_map: &mut PartNodeMap,
    options: &CadTessellationOptions,
) -> Result<()> {
    // Compute result before mutating — if tessellation fails, nothing is changed.
    let computed = compute_boolean(kind, target, tools, document, part_map)?;
    let new_node = tessellate_into(&computed.shape, scene, options, None, Some("Boolean result"))
        .context("Failed to tessellate boolean result")?;

    // Tessellation succeeded — commit mutations.
    for (&node, &part_id) in tools.iter().zip(computed.tool_part_ids.iter()) {
        scene.remove_node(node);
        document.remove_part(part_id);
        part_map.remove_by_part(part_id);
    }
    scene.remove_node(target);
    document.remove_part(computed.target_part_id);
    part_map.remove_by_part(computed.target_part_id);

    let new_part_id = document.add_part(
        "Boolean result".to_owned(),
        computed.shape,
        duck_engine_viewer::common::Transform::IDENTITY,
        computed.color,
    );
    part_map.insert(new_part_id, new_node);

    Ok(())
}

/// Non-destructive preview: compute the boolean and add a temporary scene node
/// without modifying the source parts or document. The caller owns the returned
/// NodeId and must remove it when done.
pub fn preview_boolean(
    kind: BooleanKind,
    target: NodeId,
    tools: &[NodeId],
    scene: &mut Scene,
    document: &CadDocument,
    part_map: &PartNodeMap,
    options: &CadTessellationOptions,
) -> Result<NodeId> {
    let computed = compute_boolean(kind, target, tools, document, part_map)?;
    tessellate_into(&computed.shape, scene, options, None, Some("Boolean preview"))
        .context("Failed to tessellate boolean preview")
}
