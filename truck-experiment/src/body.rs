use std::collections::{HashMap, HashSet};

use truck_modeling as truck;

use crate::edge::{Edge, EdgeId};
use crate::face::{Face, FaceId};

pub type BodyId = u32;

/// A CAD body — a collection of faces and edges forming a shell or solid.
///
/// Constructed from a Truck `Shell` (or the shell of a `Solid`). The body
/// assigns stable IDs to each face and edge for selection and mapping.
pub struct Body {
    id: BodyId,
    faces: HashMap<FaceId, Face>,
    edges: HashMap<EdgeId, Edge>,
    next_face_id: FaceId,
    next_edge_id: EdgeId,
}

impl Body {
    /// Create a body from a Truck shell, assigning IDs to all faces and edges.
    pub fn from_truck_shell(id: BodyId, shell: &truck::Shell) -> Self {
        let mut body = Self {
            id,
            faces: HashMap::new(),
            edges: HashMap::new(),
            next_face_id: 0,
            next_edge_id: 0,
        };

        // Track Truck edge IDs (as u64 hash) we've already added to avoid duplicates
        // (edges are shared between adjacent faces).
        let mut seen_edges: HashSet<u64> = HashSet::new();

        for truck_face in shell.face_iter() {
            let face_id = body.next_face_id;
            body.next_face_id += 1;
            body.faces.insert(face_id, Face::new(face_id, truck_face.clone()));

            // Extract edges from the face's boundary wires
            for wire in truck_face.boundaries() {
                for truck_edge in wire.edge_iter() {
                    // Use the Truck ID's hash as a unique identifier
                    let id_hash = {
                        use std::hash::{Hash, Hasher};
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        truck_edge.id().hash(&mut hasher);
                        hasher.finish()
                    };
                    if !seen_edges.insert(id_hash) {
                        continue;
                    }
                    let edge_id = body.next_edge_id;
                    body.next_edge_id += 1;
                    body.edges.insert(edge_id, Edge::new(edge_id, truck_edge.clone()));
                }
            }
        }

        body
    }

    /// Create a body from a Truck solid (uses its outer shell).
    pub fn from_truck_solid(id: BodyId, solid: &truck::Solid) -> Self {
        Self::from_truck_shell(id, solid.boundaries().first().expect("solid must have at least one shell"))
    }

    pub fn id(&self) -> BodyId {
        self.id
    }

    pub fn faces(&self) -> &HashMap<FaceId, Face> {
        &self.faces
    }

    pub fn edges(&self) -> &HashMap<EdgeId, Edge> {
        &self.edges
    }

    pub fn face(&self, id: FaceId) -> Option<&Face> {
        self.faces.get(&id)
    }

    pub fn edge(&self, id: EdgeId) -> Option<&Edge> {
        self.edges.get(&id)
    }
}
