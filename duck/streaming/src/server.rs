use std::net::{SocketAddr, TcpListener};
use std::sync::{Arc, Mutex, MutexGuard};
use std::sync::mpsc::{self, SyncSender};
use std::time::Duration;

use duck_engine_scene::{Scene, SceneEvent};

use crate::codec::{CameraHint, ClientMessage, SequencedEvent, ServerMessage};
use crate::priority::build_priority_queue;
use crate::transport::{ServerChannel, TcpByteChannel};

/// Channel used to push live event batches to one connected client.
/// Bounded so a slow client cannot exhaust server memory.
type LiveSender = SyncSender<Vec<SequencedEvent>>;

/// Streaming server that holds the authoritative scene and accepts client connections.
///
/// Clients receive an initial priority-ordered sync followed by live mutation events.
/// The server is cheaply cloneable via `Arc`; call `serve` on the arc from a background thread.
pub struct StreamingServer {
    addr: SocketAddr,
    scene: Arc<Mutex<Scene>>,
    /// One sender per connected client. Entries are pruned when the client disconnects.
    live_senders: Arc<Mutex<Vec<LiveSender>>>,
}

impl StreamingServer {
    /// Create a server that will listen on `addr`.
    /// `log_capacity` controls how many events the scene retains for reconnect fast-paths.
    pub fn new(addr: SocketAddr, mut scene: Scene, log_capacity: usize) -> Self {
        scene.enable_event_log(log_capacity);
        Self {
            addr,
            scene: Arc::new(Mutex::new(scene)),
            live_senders: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Apply a mutation to the authoritative scene and broadcast it to all connected clients.
    pub fn apply_event(&self, event: SceneEvent) {
        let seq = {
            let mut scene = self.scene.lock().unwrap();
            // apply_event_to_scene calls Scene's instrumented mutation methods,
            // which push the event into the embedded SceneEventLog automatically.
            crate::apply::apply_event_to_scene(&mut scene, event.clone());
            scene.event_log().map(|log| log.next_seq().saturating_sub(1)).unwrap_or(0)
        };
        broadcast(&self.live_senders, vec![SequencedEvent { seq, event }]);
    }

    /// Start the TCP accept loop. Blocks the calling thread; run this in a dedicated thread.
    pub fn serve(self: Arc<Self>) {
        let listener = TcpListener::bind(self.addr).expect("bind failed");
        log::info!("Streaming server listening on {}", self.addr);

        let spin_out_connection_handler = |server, stream| {
            std::thread::spawn(move || {
                let channel = TcpByteChannel::new(stream);
                if let Err(e) = handle_connection(channel, server) {
                    log::warn!("Client disconnected: {e:#}");
                }
            });
        };

        // Listen for incoming connections. Then spin out a thread to start serving
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let server = Arc::clone(&self);
                    spin_out_connection_handler(server, stream);
                }
                Err(e) => log::error!("Accept error: {e}"),
            }
        }
    }
}

/// Entry point for one client connection. Runs entirely on its own thread.
fn handle_connection<C: ServerChannel>(
    mut channel: C,
    server: Arc<StreamingServer>,
) -> anyhow::Result<()> {
    let ClientMessage::Subscribe(opts) = channel.recv_client()?
    else {
        anyhow::bail!("expected Subscribe as first message");
    };

    // Register a bounded channel so the server can push live events to this client.
    let (live_tx, live_rx) = mpsc::sync_channel::<Vec<SequencedEvent>>(64);
    server.live_senders.lock().unwrap().push(live_tx);

    // Attempt reconnect fast-path first; fall back to a full priority sync if the
    // log is too stale or this is a fresh connect (client_seq == 0).
    let fast_path = try_reconnect_fast_path(&mut channel, &server.scene, opts.client_seq)?;
    if !fast_path {
        run_priority_sync(&mut channel, &server.scene, opts.camera.as_ref())?;
    }

    // Both paths end here: the client is now in live mode.
    return live_update_loop(&mut channel, &live_rx);
}

/// Send delta events accumulated since `client_seq`. Returns `true` if the fast-path was taken.
/// Returns `false` when the log is too stale or this is a fresh connect, signalling that the
/// caller should fall back to a full priority sync.
fn try_reconnect_fast_path<C: ServerChannel>(
    channel: &mut C,
    scene: &Mutex<Scene>,
    client_seq: u64,
) -> anyhow::Result<bool> {
    if client_seq == 0 {
        return Ok(false);
    }
    let scene = scene.lock().unwrap();
    let Some(log) = scene.event_log() else {
        return Ok(false);
    };
    let Some(iter) = log.events_since(client_seq) else {
        return Ok(false); // log too stale; caller will run a full sync
    };
    let events: Vec<SequencedEvent> = iter.cloned().collect();
    let server_seq = log.next_seq().saturating_sub(1);
    drop(scene);

    channel.send_server(&ServerMessage::EventBatch { events })?;
    channel.send_server(&ServerMessage::SyncComplete { server_seq })?;
    Ok(true)
}

/// Stream all scene resources to a fresh client in priority order, then send SyncComplete.
///
/// Resources are sent highest-priority first (nodes → instances → materials → meshes by
/// screen coverage → textures → environment maps) so the client can start rendering
/// before the full scene has arrived.
fn run_priority_sync<C: ServerChannel>(
    channel: &mut C,
    scene: &Mutex<Scene>,
    camera: Option<&CameraHint>,
) -> anyhow::Result<()> {
    let (priority_resources, server_seq) = {
        let scene = scene.lock().unwrap();
        let queue = build_priority_queue(&scene, camera);
        let seq = scene.event_log().map(|l| l.next_seq().saturating_sub(1)).unwrap_or(0);
        (queue, seq)
    };

    let mut paused = false;
    for pending in priority_resources {
        // Non-blocking poll for flow-control messages between resource sends.
        match channel.try_recv_client()? {
            Some(ClientMessage::Pause) => paused = true,
            Some(ClientMessage::Resume) => paused = false,
            Some(ClientMessage::CameraUpdate(_)) => { 
                todo!("Mid-stream camera update")
            }
            _ => {}
        }
        if paused {
            // Block until the client sends Resume.
            loop {
                if let ClientMessage::Resume = channel.recv_client()? {
                    paused = false;
                    break;
                }
            }
        }

        let event = {
            let scene = scene.lock().unwrap();
            crate::apply::build_resource_event(&scene, pending.kind, pending.id)
        };
        let Some(event) = event else { continue };

        channel.send_server(&ServerMessage::EventBatch {
            events: vec![SequencedEvent { seq: server_seq, event }],
        })?;
    }

    channel.send_server(&ServerMessage::SyncComplete { server_seq })?;
    Ok(())
}

/// Forward live event batches from the server to the client indefinitely.
/// Sends Ping on 30-second idle to keep the connection alive.
fn live_update_loop<C: ServerChannel>(
    channel: &mut C,
    live_rx: &mpsc::Receiver<Vec<SequencedEvent>>,
) -> anyhow::Result<()> {
    loop {
        match live_rx.recv_timeout(Duration::from_secs(30)) {
            Ok(batch) if batch.is_empty() => {
                // Empty batch is a sentinel from the server indicating shutdown.
                channel.send_server(&ServerMessage::Goodbye)?;
                return Ok(());
            }
            Ok(batch) => {
                channel.send_server(&ServerMessage::EventBatch { events: batch })?;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                channel.send_server(&ServerMessage::Ping)?;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                channel.send_server(&ServerMessage::Goodbye)?;
                return Ok(());
            }
        }
        // Non-blocking drain of any pending client control messages.
        if let Some(ClientMessage::Pause) = channel.try_recv_client()? {
            loop {
                if let ClientMessage::Resume = channel.recv_client()? {
                    break;
                }
            }
        }
    }
}

/// Send `batch` to every registered client. Prunes senders whose receivers have dropped.
fn broadcast(senders: &Mutex<Vec<LiveSender>>, batch: Vec<SequencedEvent>) {
    let mut guard: MutexGuard<Vec<LiveSender>> = senders.lock().unwrap();
    guard.retain(|s| s.try_send(batch.clone()).is_ok());
}
