//! Platform bootstrap: event-loop creation, the entry point, and per-frame
//! window/viewer-state initialization.
//!
//! `run()` is the body of `main`; `resume()` is called from
//! `ApplicationHandler::resumed`. Native builds the viewer synchronously;
//! web creates the canvas, then builds the viewer asynchronously and delivers
//! it back through the event-loop proxy.

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native::{resume, run};

#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(target_arch = "wasm32")]
pub(crate) use web::{resume, run};
