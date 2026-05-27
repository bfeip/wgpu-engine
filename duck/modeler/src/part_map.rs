use std::collections::HashMap;

use duck_engine_scene::NodeId;

use crate::document::PartId;

pub struct PartNodeMap {
    part_to_node: HashMap<PartId, NodeId>,
    node_to_part: HashMap<NodeId, PartId>,
}

impl PartNodeMap {
    pub fn new() -> Self {
        Self {
            part_to_node: HashMap::new(),
            node_to_part: HashMap::new(),
        }
    }

    pub fn insert(&mut self, part: PartId, node: NodeId) {
        self.part_to_node.insert(part, node);
        self.node_to_part.insert(node, part);
    }

    pub fn remove_by_part(&mut self, part: PartId) -> Option<NodeId> {
        if let Some(node) = self.part_to_node.remove(&part) {
            self.node_to_part.remove(&node);
            Some(node)
        } else {
            None
        }
    }

    pub fn remove_by_node(&mut self, node: NodeId) -> Option<PartId> {
        if let Some(part) = self.node_to_part.remove(&node) {
            self.part_to_node.remove(&part);
            Some(part)
        } else {
            None
        }
    }

    pub fn node_for_part(&self, part: PartId) -> Option<NodeId> {
        self.part_to_node.get(&part).copied()
    }

    pub fn part_for_node(&self, node: NodeId) -> Option<PartId> {
        self.node_to_part.get(&node).copied()
    }
}
