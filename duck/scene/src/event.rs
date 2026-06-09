use std::collections::VecDeque;

use crate::{
    common::Transform,
    EnvironmentMap, EnvironmentMapId,
    FaceMaterial, FaceMaterialId,
    Instance, InstanceId,
    LineMaterial, LineMaterialId,
    Mesh, MeshId,
    PointMaterial, PointMaterialId,
    Node, NodeId,
    NodePayload,
    Texture, TextureId,
    Visibility,
};

/// A discrete mutation to the scene, sufficient to reconstruct the change on a remote client.
///
/// `NodePayload::Custom` is already `#[serde(skip)]`, so custom payloads degrade to
/// `NodePayload::None` over the wire.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SceneEvent {
    MeshAdded(MeshId, Mesh),
    MeshRemoved(MeshId),

    FaceMaterialAdded(FaceMaterialId, FaceMaterial),
    FaceMaterialRemoved(FaceMaterialId),
    LineMaterialAdded(LineMaterialId, LineMaterial),
    LineMaterialRemoved(LineMaterialId),
    PointMaterialAdded(PointMaterialId, PointMaterial),
    PointMaterialRemoved(PointMaterialId),

    TextureAdded(TextureId, Texture),

    InstanceAdded(InstanceId, Instance),
    InstanceRemoved(InstanceId),

    /// Carries the full Node so the client can call `Scene::insert_node`, preserving
    /// the server-assigned UUID v7.
    NodeAdded(Node),
    NodeRemoved(NodeId),
    NodeTransformSet(NodeId, Transform),
    NodePayloadSet(NodeId, NodePayload),
    NodeVisibilitySet(NodeId, Visibility),

    EnvironmentMapAdded(EnvironmentMapId, EnvironmentMap),
    ActiveEnvironmentMapSet(Option<EnvironmentMapId>),

    ActiveCameraSet(Option<NodeId>),
}

/// A scene event paired with its monotonic sequence number from a [`SceneEventLog`].
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SequencedEvent {
    pub seq: u64,
    pub event: SceneEvent,
}

/// Ring buffer of sequenced scene events.
///
/// Attached to a `Scene` via `Scene::enable_event_log`. When `None`, all mutation
/// methods run with zero extra overhead. When `Some`, each mutation appends an event.
///
/// The sequence number is a monotonically increasing `u64` internal to the log; it is
/// unrelated to `node_generation`. Streaming clients track their `last_seq` and use it
/// to request delta updates on reconnect.
pub struct SceneEventLog {
    entries: VecDeque<SequencedEvent>,
    next_seq: u64,
    capacity: usize,
}

impl SceneEventLog {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            next_seq: 0,
            capacity,
        }
    }

    /// Append an event. Drops the oldest entry when at capacity.
    pub fn push(&mut self, event: SceneEvent) {
        let seq = self.next_seq;
        self.next_seq += 1;
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(SequencedEvent { seq, event });
    }

    /// Sequence number that will be assigned to the next event (i.e., current "head").
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// Returns an iterator over events with sequence number strictly greater than `after_seq`.
    ///
    /// Returns `None` if `after_seq` is older than the oldest retained event, signalling the
    /// caller that a full re-sync is needed instead of a delta.
    pub fn events_since(&self, after_seq: u64) -> Option<impl Iterator<Item = &SequencedEvent>> {
        if let Some(oldest) = self.entries.front() {
            if after_seq < oldest.seq {
                return None;
            }
        }
        Some(self.entries.iter().filter(move |e| e.seq > after_seq))
    }

    /// Oldest retained sequence number, or `None` if the log is empty.
    pub fn oldest_seq(&self) -> Option<u64> {
        self.entries.front().map(|e| e.seq)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
