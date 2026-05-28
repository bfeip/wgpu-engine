use std::sync::{Arc, Mutex};

use duck_engine_common::RgbaColor;
use duck_engine_scene::{Id, Scene};
use duck_engine_viewer::common::Transform;
use opencascade::primitives::Shape;

use crate::part_map::PartNodeMap;

pub type PartId = Id;

pub struct CadPart {
    pub id: PartId,
    pub name: String,
    pub shape: Shape,
    pub transform: Transform,
    pub color: RgbaColor,
    pub visible: bool,
}

pub struct Document {
    parts: Vec<CadPart>,
    pub part_map: PartNodeMap,
    pub scene: Arc<Mutex<Scene>>,
}

impl Document {
    pub fn new(scene: Arc<Mutex<Scene>>) -> Self {
        Self {
            parts: Vec::new(),
            part_map: PartNodeMap::new(),
            scene,
        }
    }

    pub fn add_part(
        &mut self,
        name: String,
        shape: Shape,
        transform: Transform,
        color: RgbaColor,
    ) -> PartId {
        let id = PartId::new();
        self.parts.push(CadPart { id, name, shape, transform, color, visible: true });
        id
    }

    pub fn remove_part(&mut self, id: PartId) {
        self.parts.retain(|p| p.id != id);
    }

    pub fn get_part(&self, id: PartId) -> Option<&CadPart> {
        self.parts.iter().find(|p| p.id == id)
    }

    pub fn get_part_mut(&mut self, id: PartId) -> Option<&mut CadPart> {
        self.parts.iter_mut().find(|p| p.id == id)
    }

    pub fn parts(&self) -> impl Iterator<Item = &CadPart> {
        self.parts.iter()
    }
}
