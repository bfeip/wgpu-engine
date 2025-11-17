use std::collections::HashMap;

use wgpu::ShaderModuleDescriptor;

use crate::material::MaterialType;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum ShaderFragment {
    Common,
    Vertex,
    FaceColorFragment,
    FaceTextureFragment,
    LineColorFragment,
    PointColorFragment,
}

pub struct ShaderBuilder {
    fragments: HashMap<ShaderFragment, &'static str>
}

impl ShaderBuilder {
    pub fn new() -> Self {
        let mut fragments = HashMap::with_capacity(6);
        fragments.insert(ShaderFragment::Common, include_str!("shaders/common.wgsl"));
        fragments.insert(ShaderFragment::Vertex, include_str!("shaders/vertex.wgsl"));
        fragments.insert(ShaderFragment::FaceColorFragment, include_str!("shaders/face_color_fragment.wgsl"));
        fragments.insert(ShaderFragment::FaceTextureFragment, include_str!("shaders/face_texture_fragment.wgsl"));
        fragments.insert(ShaderFragment::LineColorFragment, include_str!("shaders/line_color_fragment.wgsl"));
        fragments.insert(ShaderFragment::PointColorFragment, include_str!("shaders/point_color_fragment.wgsl"));
        Self {
            fragments
        }
    }

    pub fn generate_shader(&self, device: &wgpu::Device, material_type: MaterialType) -> wgpu::ShaderModule {
        let shader_preamble = self.fragments[&ShaderFragment::Common].chars().chain(
            self.fragments[&ShaderFragment::Vertex].chars()
        );

        let shader_source_string: String = match material_type {
            MaterialType::FaceColor => {
                shader_preamble.chain(
                    self.fragments[&ShaderFragment::FaceColorFragment].chars()
                ).collect()
            },
            MaterialType::FaceTexture => {
                shader_preamble.chain(
                    self.fragments[&ShaderFragment::FaceTextureFragment].chars()
                ).collect()
            },
            MaterialType::LineColor => {
                shader_preamble.chain(
                    self.fragments[&ShaderFragment::LineColorFragment].chars()
                ).collect()
            },
            MaterialType::PointColor => {
                shader_preamble.chain(
                    self.fragments[&ShaderFragment::PointColorFragment].chars()
                ).collect()
            },
        };

        let shader_label: &str = match material_type {
            MaterialType::FaceColor => "Face Color Material Shader",
            MaterialType::FaceTexture => "Face Texture Material Shader",
            MaterialType::LineColor => "Line Color Material Shader",
            MaterialType::PointColor => "Point Color Material Shader",
        };

        device.create_shader_module(ShaderModuleDescriptor {
            label: Some(shader_label),
            source: wgpu::ShaderSource::Wgsl(shader_source_string.into())
        })
    }
}