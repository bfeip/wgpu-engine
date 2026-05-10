use duck_engine_scene::Scene;
use duck_engine_streaming::{apply_event_to_scene, CameraHint, SceneUpdate, StreamingClient, SubscribeOptions};

pub enum PollResult {
    /// At least one event was applied to the scene this frame.
    Events,
    /// Initial sync completed this frame.
    SyncComplete,
    /// The server disconnected or an I/O error occurred.
    Disconnected,
    /// No update was available.
    Idle,
}

/// Thin wrapper around `StreamingClient` that integrates with the `Viewer`'s scene.
pub struct ViewerStreamClient {
    client: StreamingClient,
    /// Set to `true` after `SyncComplete` is received.
    pub sync_complete: bool,
}

impl ViewerStreamClient {
    /// Connect to `addr` and begin streaming with the given options.
    pub fn connect(addr: &str, opts: SubscribeOptions) -> anyhow::Result<Self> {
        let client = StreamingClient::connect(addr, opts)?;
        Ok(Self { client, sync_complete: false })
    }

    /// Drain all pending updates and apply them to `scene`. Returns the most significant
    /// result from this frame: `SyncComplete` > `Events` > `Disconnected` > `Idle`.
    pub fn poll(&mut self, scene: &mut Scene) -> PollResult {
        let mut result = PollResult::Idle;
        loop {
            match self.client.poll() {
                Some(SceneUpdate::Events(events)) => {
                    for se in events {
                        apply_event_to_scene(scene, se.event);
                    }
                    if !matches!(result, PollResult::SyncComplete) {
                        result = PollResult::Events;
                    }
                }
                Some(SceneUpdate::SyncComplete { .. }) => {
                    self.sync_complete = true;
                    result = PollResult::SyncComplete;
                }
                Some(SceneUpdate::Disconnected) => {
                    return PollResult::Disconnected;
                }
                None => return result,
            }
        }
    }

    /// Notify the server of a camera position change for priority re-sorting.
    pub fn update_camera(&self, hint: CameraHint) {
        self.client.update_camera(hint);
    }
}
