use std::path::PathBuf;

use criterion::{Criterion, criterion_group, criterion_main};
use duck_engine_import_export::{LoadOptions, SceneSource, format, load_sync};

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets")
}

fn glb_path() -> PathBuf {
    assets_dir().join("1987_mazda_rx-7_fc.glb")
}

fn gltf_path() -> PathBuf {
    assets_dir().join("Camera_01_4k.gltf/Camera_01_4k.gltf")
}

fn bench_gltf_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("gltf_loading");

    group.bench_function("load_glb", |b| {
        b.iter(|| {
            load_sync(
                SceneSource::Path(glb_path()),
                LoadOptions::default(),
            )
            .unwrap()
        });
    });

    group.bench_function("load_gltf_with_textures", |b| {
        b.iter(|| {
            load_sync(
                SceneSource::Path(gltf_path()),
                LoadOptions::default(),
            )
            .unwrap()
        });
    });

    group.finish();
}

fn bench_duck_serialization(c: &mut Criterion) {
    // Pre-load a scene from the GLB file for duck benchmarks
    let result = load_sync(
        SceneSource::Path(glb_path()),
        LoadOptions::default(),
    )
    .unwrap();
    let scene = result.scene;
    let duck_bytes = format::to_bytes(&scene).unwrap();

    let mut group = c.benchmark_group("duck_serialization");

    group.bench_function("to_bytes", |b| {
        b.iter(|| format::to_bytes(&scene).unwrap());
    });

    group.bench_function("from_bytes", |b| {
        b.iter(|| format::from_bytes(&duck_bytes).unwrap());
    });

    group.finish();
}

fn bench_duck_file_io(c: &mut Criterion) {
    let result = load_sync(
        SceneSource::Path(glb_path()),
        LoadOptions::default(),
    )
    .unwrap();
    let scene = result.scene;

    // Write a Duck file once for load benchmarks
    let tmp_dir = std::env::temp_dir().join("duck_engine_bench");
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let duck_path = tmp_dir.join("bench_scene.duck");
    format::save_to_file(&scene, &duck_path).unwrap();

    let mut group = c.benchmark_group("duck_file_io");

    group.bench_function("save_to_file", |b| {
        let save_path = tmp_dir.join("bench_save.duck");
        b.iter(|| format::save_to_file(&scene, &save_path).unwrap());
    });

    group.bench_function("load_from_file", |b| {
        b.iter(|| format::load_from_file(&duck_path).unwrap());
    });

    group.finish();

    // Clean up
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

// ============================================================================
// Criterion Harness
// ============================================================================

criterion_group!(
    benches,
    bench_gltf_loading,
    bench_duck_serialization,
    bench_duck_file_io,
);
criterion_main!(benches);
