use std::collections::HashMap;

use wgpu::ShaderModuleDescriptor;

use crate::material::MaterialType;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum ShaderFragment {
    Common,
    Vertex,
    ColorFragment,
    TextureFragment,
}

pub struct ShaderBuilder {
    fragments: HashMap<ShaderFragment, &'static str>
}

impl ShaderBuilder {
    pub fn new() -> Self {
        let mut fragments = HashMap::with_capacity(4);
        fragments.insert(ShaderFragment::Common, include_str!("shaders/common.wgsl"));
        fragments.insert(ShaderFragment::Vertex, include_str!("shaders/vertex.wgsl"));
        fragments.insert(ShaderFragment::ColorFragment, include_str!("shaders/color_fragment.wgsl"));
        fragments.insert(ShaderFragment::TextureFragment, include_str!("shaders/texture_fragment.wgsl"));
        Self {
            fragments
        }
    }

    pub fn generate_shader(&self, device: &wgpu::Device, material_type: MaterialType) -> wgpu::ShaderModule {
        let shader_preamble = self.fragments[&ShaderFragment::Common].chars().chain(
            self.fragments[&ShaderFragment::Vertex].chars()
        );

        let shader_source_string: String = match material_type {
            MaterialType::Color => {
                shader_preamble.chain(
                    self.fragments[&ShaderFragment::ColorFragment].chars()
                ).collect()
            },
            MaterialType::Texture => {
                shader_preamble.chain(
                    self.fragments[&ShaderFragment::TextureFragment].chars()
                ).collect()
            }
        };

        let shader_label: &str = match material_type {
            MaterialType::Color => "Color Material Shader",
            MaterialType::Texture => "Texture Material Shader"
        };

        device.create_shader_module(ShaderModuleDescriptor {
            label: Some(shader_label),
            source: wgpu::ShaderSource::Wgsl(shader_source_string.into())
        })
    }
}