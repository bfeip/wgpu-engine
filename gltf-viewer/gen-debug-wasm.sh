#! /bin/bash
cargo build --target "wasm32-unknown-unknown" &&
wasm-bindgen --debug --keep-debug --target web --out-dir ../target/wasm-bindgen ../target/wasm32-unknown-unknown/debug/gltf-viewer.wasm &&
echo "bindgen finished" &&
cp -v ../target/wasm-bindgen/gltf-viewer_bg.wasm ./dist/