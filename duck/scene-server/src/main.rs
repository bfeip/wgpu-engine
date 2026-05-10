use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use duck_engine_import_export::{load_sync, SceneSource};
use duck_engine_streaming::StreamingServer;

#[derive(Parser)]
#[command(about = "Stream a scene file to connected viewers over TCP")]
struct Args {
    /// Scene file to load and serve (DUCK, glTF, etc.)
    file: PathBuf,

    /// TCP port to listen on
    #[arg(long, default_value = "7878")]
    port: u16,

    /// Number of past events to retain for reconnect fast-paths
    #[arg(long, default_value = "2000")]
    log_capacity: usize,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    log::info!("Loading {:?}…", args.file);
    let loaded = load_sync(SceneSource::Path(args.file), Default::default())
        .context("load scene")?;
    log::info!("Scene loaded: {} meshes", loaded.scene.mesh_count());

    let addr: SocketAddr = ([0, 0, 0, 0], args.port).into();
    let server = Arc::new(StreamingServer::new(addr, loaded.scene, args.log_capacity));

    let srv = Arc::clone(&server);
    std::thread::spawn(move || srv.serve());

    log::info!("Listening on {addr}");
    loop {
        std::thread::sleep(Duration::from_secs(60));
    }
}
