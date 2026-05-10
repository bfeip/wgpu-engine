use std::io::{Read, Write};

use anyhow::Context;
use duck_engine_scene::NodeId;
use serde::{Deserialize, Serialize};

pub use duck_engine_scene::SequencedEvent;

/// Camera position/orientation sent from client to server for priority sorting.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CameraHint {
    pub position: [f32; 3],
    pub forward: [f32; 3],
    pub fov_y_rad: f32,
}

/// Options passed to the server when subscribing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeOptions {
    /// `0` for a fresh connect. A non-zero value requests a delta from that sequence number,
    /// enabling the reconnect fast-path when the server's event log is still warm.
    pub client_seq: u64,
    /// Subscribe only to a subtree rooted at this node. `None` = entire scene.
    pub root_node: Option<NodeId>,
    /// Initial camera hint used by the server to sort resource priority during initial sync.
    pub camera: Option<CameraHint>,
}

impl Default for SubscribeOptions {
    fn default() -> Self {
        Self { client_seq: 0, root_node: None, camera: None }
    }
}

/// Messages sent from client to server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientMessage {
    /// Initial subscription. `client_seq = 0` for a fresh connect; higher values
    /// request a delta from that sequence number (reconnect fast-path).
    Subscribe(SubscribeOptions),
    /// Update server-side priority sorting while streaming is in progress.
    CameraUpdate(CameraHint),
    /// Signal that the client's receive buffer is full — server should pause data.
    Pause,
    /// Resume sending after a Pause.
    Resume,
}

/// Messages sent from server to client.
#[derive(Clone, Serialize, Deserialize)]
pub enum ServerMessage {
    /// One or more sequenced events applied in order.
    /// Used for both initial sync and live mutation delta updates.
    EventBatch { events: Vec<SequencedEvent> },
    /// All initial resources have been streamed; client is now in live mode.
    SyncComplete { server_seq: u64 },
    Ping,
    Goodbye,
}

// Framing: [4-byte u32 LE payload length][zstd-compressed bincode payload]

pub fn encode<T: Serialize>(msg: &T) -> anyhow::Result<Vec<u8>> {
    let raw = bincode::serde::encode_to_vec(msg, bincode::config::standard())
        .context("bincode encode")?;
    let compressed = zstd::encode_all(std::io::Cursor::new(&raw), 3)
        .context("zstd compress")?;
    let len = compressed.len() as u32;
    let mut out = Vec::with_capacity(4 + compressed.len());
    out.write_all(&len.to_le_bytes()).unwrap();
    out.extend_from_slice(&compressed);
    Ok(out)
}

pub fn decode_from_reader<T: for<'de> Deserialize<'de>>(reader: &mut impl Read) -> anyhow::Result<T> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf).context("read frame length")?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload).context("read frame payload")?;
    decode_payload(&payload)
}

fn decode_payload<T: for<'de> Deserialize<'de>>(payload: &[u8]) -> anyhow::Result<T> {
    let decompressed = zstd::decode_all(std::io::Cursor::new(payload))
        .context("zstd decompress")?;
    let (msg, _) = bincode::serde::decode_from_slice(&decompressed, bincode::config::standard())
        .context("bincode decode")?;
    Ok(msg)
}