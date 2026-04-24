// Gooch shading shader
//
// Standalone WGSL — struct layouts must match the renderer's GPU-side types:
//   Camera        ↔ CameraUniform        (80 bytes)
//   LightUniform  ↔ LightUniform         (64 bytes)
//   LightsUniform ↔ LightsArrayUniform   (528 bytes)
//   VertexInput   ↔ Vertex               (36 bytes: position + tex_coords + normal, all vec3)
//   InstanceInput ↔ GpuInstance          (4×vec4 model columns + 3×vec3 normal columns)

struct Camera {
    view_proj: mat4x4<f32>,
    eye_position: vec3<f32>,
    // 4 bytes implicit padding to reach 80 bytes total (WGSL aligns struct to 16)
};
@group(0) @binding(0) var<uniform> camera: Camera;

const LIGHT_TYPE_DIRECTIONAL: u32 = 1u;

struct LightUniform {
    light_type:     u32,
    range:          f32,
    inner_cone_cos: f32,
    outer_cone_cos: f32,
    position:       vec3<f32>,
    intensity:      f32,
    direction:      vec3<f32>,
    _padding1:      f32,
    color:          vec3<f32>,
    _padding2:      f32,
};

struct LightsUniform {
    light_count: u32,
    // WGSL inserts 12 bytes padding before the array to align it to 16 bytes.
    // The Rust struct has explicit [u32; 3] padding to match this layout.
    lights: array<LightUniform, 8>,
};
@group(1) @binding(0) var<uniform> lights: LightsUniform;

// ── Vertex / instance inputs ──────────────────────────────────────────────────

struct VertexInput {
    @location(0) position:  vec3<f32>,
    @location(1) tex_coords: vec3<f32>, // third component unused; kept to match Vertex layout
    @location(2) normal:    vec3<f32>,
};

struct InstanceInput {
    // Model matrix columns (column-major, matches GpuInstance::transform)
    @location(3) model_col0: vec4<f32>,
    @location(4) model_col1: vec4<f32>,
    @location(5) model_col2: vec4<f32>,
    @location(6) model_col3: vec4<f32>,
    // Normal matrix columns (column-major, matches GpuInstance::normal_mat)
    @location(7) normal_col0: vec3<f32>,
    @location(8) normal_col1: vec3<f32>,
    @location(9) normal_col2: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal:   vec3<f32>,
    @location(1) world_position: vec3<f32>,
};

@vertex
fn vs_main(vert: VertexInput, inst: InstanceInput) -> VertexOutput {
    let model = mat4x4<f32>(inst.model_col0, inst.model_col1, inst.model_col2, inst.model_col3);
    let normal_mat = mat3x3<f32>(inst.normal_col0, inst.normal_col1, inst.normal_col2);

    let world_pos = model * vec4<f32>(vert.position, 1.0);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_pos;
    out.world_normal   = normalize(normal_mat * vert.normal);
    out.world_position = world_pos.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let warm = vec3<f32>(0.95, 0.75, 0.2);  // lit side: warm gold
    let cool = vec3<f32>(0.0,  0.2,  0.7);  // shadow side: cool blue

    // Determine light direction: use the first scene light if present,
    // otherwise fall back to a default directional light.
    var light_dir = normalize(vec3<f32>(0.5, 1.0, 0.5));
    if lights.light_count > 0u {
        let L = lights.lights[0];
        if L.light_type == LIGHT_TYPE_DIRECTIONAL {
            light_dir = normalize(-L.direction);
        } else {
            light_dir = normalize(L.position - in.world_position);
        }
    }

    // Gooch interpolation: remap n·l from [-1, 1] → [0, 1]
    let n_dot_l = dot(in.world_normal, light_dir);
    let t = (n_dot_l + 1.0) * 0.5;
    let gooch = mix(cool, warm, t);

    // Silhouette darkening: fade toward the cool color at grazing angles
    let view_dir = normalize(camera.eye_position - in.world_position);
    let n_dot_v  = max(dot(in.world_normal, view_dir), 0.0);
    let edge     = smoothstep(0.0, 0.35, n_dot_v);
    let color    = mix(cool * 0.4, gooch, edge);

    return vec4<f32>(color, 1.0);
}
