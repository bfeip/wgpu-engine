# WGPU-Engine

This is my project graphics engine written in Rust with WGPU. It's currently in
active development as of November 2025.

## Current features
- Scene from glTF
- Mouse movement
- Basic color / texture shading

## Next features
- Selection and scene interaction
- Line and point rendering
- NURBS

## Building and testing

### Native

**Build:** `cargo build`

**Test:** `cargo test`

### Web

Requires wasm-pack

**Build:** `wasm-pack build`