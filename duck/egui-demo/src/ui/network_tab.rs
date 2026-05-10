use duck_engine_viewer::Viewer;

use super::UiActions;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkStatus {
    Disconnected,
    Connected,
    Error(String),
}

impl Default for NetworkStatus {
    fn default() -> Self {
        Self::Disconnected
    }
}

#[derive(Default)]
pub struct NetworkTabState {
    pub url: String,
    pub status: NetworkStatus,
}

pub fn show(ui: &mut egui::Ui, viewer: &mut Viewer, actions: &mut UiActions, state: &mut NetworkTabState) {
    if state.url.is_empty() {
        state.url = "127.0.0.1:7878".to_string();
    }

    ui.label("Server address:");
    ui.text_edit_singleline(&mut state.url);
    ui.add_space(4.0);

    let connected = matches!(state.status, NetworkStatus::Connected);

    if connected {
        if ui.button("Disconnect").clicked() {
            actions.disconnect_stream = true;
            state.status = NetworkStatus::Disconnected;
        }

        let sync_complete = viewer.stream_sync_complete();
        if sync_complete {
            ui.colored_label(egui::Color32::GREEN, "Connected — synced");
        } else {
            ui.colored_label(egui::Color32::YELLOW, "Streaming…");
            // Indeterminate progress bar while sync is in progress.
            let t = ui.ctx().input(|i| i.time) as f32;
            let progress = (t * 0.5).sin() * 0.5 + 0.5;
            ui.add(egui::ProgressBar::new(progress).animate(true));
            ui.ctx().request_repaint();
        }
    } else {
        if ui.button("Connect").clicked() {
            actions.connect_stream = Some(state.url.clone());
        }

        if let NetworkStatus::Error(ref msg) = state.status {
            ui.colored_label(egui::Color32::RED, msg.as_str());
        }
    }
}
