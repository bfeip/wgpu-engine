use std::net::TcpStream;

use anyhow::Context;

use crate::codec::{self, ClientMessage, ServerMessage};

/// Server-side half of a connection: sends `ServerMessage`, receives `ClientMessage`.
///
/// Implemented by the transport type used by the server's per-connection handler thread.
/// To add a new transport (e.g. WebSocket), implement this trait over the blocking API
/// provided by the chosen library, then also implement `ClientChannel` on the same type.
pub trait ServerChannel: Send + 'static {
    /// Send a message to the client. Blocks until the write completes.
    fn send_server(&mut self, msg: &ServerMessage) -> anyhow::Result<()>;
    /// Receive a message from the client. Blocks until one arrives.
    fn recv_client(&mut self) -> anyhow::Result<ClientMessage>;
    /// Non-blocking poll for a client message. Returns `Ok(None)` when none is ready.
    /// Used to check for Pause/Resume/CameraUpdate between resource sends.
    fn try_recv_client(&mut self) -> anyhow::Result<Option<ClientMessage>>;
}

/// Client-side half of a connection: sends `ClientMessage`, receives `ServerMessage`.
///
/// Implemented by the transport type used by the client's background I/O thread.
pub trait ClientChannel: Send + 'static {
    /// Send a message to the server. Blocks until the write completes.
    fn send_client(&mut self, msg: &ClientMessage) -> anyhow::Result<()>;
    /// Receive a message from the server. Blocks until one arrives.
    fn recv_server(&mut self) -> anyhow::Result<ServerMessage>;
}

/// TCP implementation backed by a length-prefixed framing codec.
/// Implements both `ServerChannel` and `ClientChannel` since a `TcpStream` is full-duplex.
pub struct TcpByteChannel {
    stream: TcpStream,
}

impl TcpByteChannel {
    pub fn new(stream: TcpStream) -> Self {
        Self { stream }
    }
}

impl ServerChannel for TcpByteChannel {
    fn send_server(&mut self, msg: &ServerMessage) -> anyhow::Result<()> {
        use std::io::Write;
        let bytes = codec::encode(msg).context("encode ServerMessage")?;
        self.stream.write_all(&bytes).context("TCP write")?;
        Ok(())
    }

    fn recv_client(&mut self) -> anyhow::Result<ClientMessage> {
        codec::decode_from_reader(&mut self.stream)
    }

    fn try_recv_client(&mut self) -> anyhow::Result<Option<ClientMessage>> {
        self.stream.set_nonblocking(true).context("set_nonblocking")?;
        let result = codec::decode_from_reader::<ClientMessage>(&mut self.stream);
        let _ = self.stream.set_nonblocking(false);
        match result {
            Ok(msg) => Ok(Some(msg)),
            Err(e) => {
                let is_would_block = e
                    .downcast_ref::<std::io::Error>()
                    .map(|io| {
                        io.kind() == std::io::ErrorKind::WouldBlock
                            || io.kind() == std::io::ErrorKind::TimedOut
                    })
                    .unwrap_or(false);
                if is_would_block { Ok(None) } else { Err(e) }
            }
        }
    }
}

impl ClientChannel for TcpByteChannel {
    fn send_client(&mut self, msg: &ClientMessage) -> anyhow::Result<()> {
        use std::io::Write;
        let bytes = codec::encode(msg).context("encode ClientMessage")?;
        self.stream.write_all(&bytes).context("TCP write")?;
        Ok(())
    }

    fn recv_server(&mut self) -> anyhow::Result<ServerMessage> {
        codec::decode_from_reader(&mut self.stream)
    }
}
