use std::sync::mpsc::{self, Receiver, SyncSender};

use anyhow::Context;

use crate::codec::{CameraHint, ClientMessage, SequencedEvent, ServerMessage, SubscribeOptions};
use crate::transport::{ClientChannel, TcpByteChannel};

/// One item produced by `StreamingClient::poll`.
pub enum SceneUpdate {
    /// A batch of sequenced events to apply to the local scene replica, in order.
    Events(Vec<SequencedEvent>),
    /// Initial sync is complete; the client is now in live mode.
    SyncComplete { server_seq: u64 },
    /// The server closed the connection or an unrecoverable I/O error occurred.
    Disconnected,
}

/// Non-blocking client that receives a scene from a streaming server.
///
/// Internally runs a background I/O thread that handles all blocking network reads.
/// The render thread calls `poll` once per frame to drain any queued updates.
pub struct StreamingClient {
    /// Delivers `SceneUpdate` values from the background I/O thread.
    rx: Receiver<SceneUpdate>,
    /// Forwards control messages (camera updates) to the background I/O thread,
    /// which relays them to the server.
    cmd_tx: SyncSender<ClientMessage>,
}

impl StreamingClient {
    /// Connect to `addr` using TCP and begin streaming. Returns as soon as the Subscribe
    /// message is sent; scene events arrive asynchronously via `poll`.
    pub fn connect(addr: &str, opts: SubscribeOptions) -> anyhow::Result<Self> {
        let stream = std::net::TcpStream::connect(addr)
            .with_context(|| format!("connect to {addr}"))?;
        let channel = TcpByteChannel::new(stream);

        let (update_tx, update_rx) = mpsc::sync_channel::<SceneUpdate>(256);
        let (cmd_tx, cmd_rx) = mpsc::sync_channel::<ClientMessage>(16);

        std::thread::spawn(move || {
            run_io_thread(channel, opts, update_tx, cmd_rx);
        });

        Ok(Self { rx: update_rx, cmd_tx })
    }

    /// Non-blocking poll. Returns `Some` if an update is ready, `None` if the queue is empty.
    /// Drain all available updates before rendering to keep the local scene in sync.
    pub fn poll(&self) -> Option<SceneUpdate> {
        self.rx.try_recv().ok()
    }

    /// Tell the server to re-sort the remaining priority queue using this camera hint.
    /// Only affects streaming order during the initial sync phase; ignored in live mode.
    pub fn update_camera(&self, hint: CameraHint) {
        let _ = self.cmd_tx.try_send(ClientMessage::CameraUpdate(hint));
    }
}

// ── Background I/O thread ─────────────────────────────────────────────────────

/// Entry point for the background thread. Logs errors and always sends `Disconnected` at exit.
fn run_io_thread<C: ClientChannel>(
    mut channel: C,
    opts: SubscribeOptions,
    update_tx: SyncSender<SceneUpdate>,
    cmd_rx: Receiver<ClientMessage>,
) {
    if let Err(e) = io_loop(&mut channel, opts, &update_tx, &cmd_rx) {
        log::warn!("Streaming I/O error: {e:#}");
    }
    let _ = update_tx.send(SceneUpdate::Disconnected);
}

/// Sends Subscribe, reads `ServerMessage` values in a loop, and forwards them as
/// `SceneUpdate` values. Checks `cmd_rx` for camera updates between message reads
/// and forwards them to the server.
fn io_loop<C: ClientChannel>(
    channel: &mut C,
    opts: SubscribeOptions,
    update_tx: &SyncSender<SceneUpdate>,
    cmd_rx: &Receiver<ClientMessage>,
) -> anyhow::Result<()> {
    channel.send_client(&ClientMessage::Subscribe(opts)).context("send Subscribe")?;

    loop {
        // Forward any pending control messages (e.g. CameraUpdate) before blocking on recv.
        while let Ok(cmd) = cmd_rx.try_recv() {
            channel.send_client(&cmd).context("send client command")?;
        }

        match channel.recv_server().context("recv ServerMessage")? {
            ServerMessage::EventBatch { events } => {
                if update_tx.send(SceneUpdate::Events(events)).is_err() {
                    return Ok(()); // StreamingClient was dropped; stop the I/O thread
                }
            }
            ServerMessage::SyncComplete { server_seq } => {
                if update_tx.send(SceneUpdate::SyncComplete { server_seq }).is_err() {
                    return Ok(());
                }
            }
            ServerMessage::Ping => {
                // Keep-alive; no action needed.
            }
            ServerMessage::Goodbye => {
                return Ok(());
            }
        }
    }
}
