use anyhow::{Context, Result};
use duck_engine_scene::NodeId;
use duck_engine_scene::cad::{CadTessellationOptions, tessellate_into};
use duck_engine_viewer::scene::Scene;

use crate::document::CadDocument;
use crate::part_map::PartNodeMap;

pub enum BooleanKind {
    Subtract,
    Union,
    Intersect,
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
    let target_part_id = part_map.part_for_node(target)
        .context("Target node is not a known CAD part")?;
    let tool_part_ids: Vec<_> = tools.iter()
        .map(|&node| part_map.part_for_node(node).context("Tool node is not a known CAD part"))
        .collect::<Result<_>>()?;

    let target_shape = document.get_part(target_part_id)
        .context("Target part not found")?
        .shape
        .clone();
    let target_color = document.get_part(target_part_id).unwrap().color;
    let tool_shapes: Vec<_> = tool_part_ids.iter()
        .map(|&id| document.get_part(id).map(|p| p.shape.clone()).context("Tool part not found"))
        .collect::<Result<_>>()?;

    // Compute result before mutating — if tessellation fails, nothing is changed.
    let mut result = target_shape;
    for tool in &tool_shapes {
        result = match kind {
            BooleanKind::Subtract  => result.subtract(tool).shape,
            BooleanKind::Union     => result.union(tool).shape,
            BooleanKind::Intersect => result.intersect(tool).shape,
        };
    }
    let new_node = tessellate_into(&result, scene, options, None, Some("Boolean result"))
        .context("Failed to tessellate boolean result")?;

    // Tessellation succeeded — commit mutations.
    for (&node, &part_id) in tools.iter().zip(tool_part_ids.iter()) {
        scene.remove_node(node);
        document.remove_part(part_id);
        part_map.remove_by_part(part_id);
    }
    scene.remove_node(target);
    document.remove_part(target_part_id);
    part_map.remove_by_part(target_part_id);

    let new_part_id = document.add_part(
        "Boolean result".to_owned(),
        result,
        duck_engine_viewer::common::Transform::IDENTITY,
        target_color,
    );
    part_map.insert(new_part_id, new_node);

    Ok(())
}
