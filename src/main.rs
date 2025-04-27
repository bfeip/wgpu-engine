use wgpu_engine::run;

fn main() {
    pollster::block_on(run());
}