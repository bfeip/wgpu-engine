use std::path::Path;

struct GltfParser {

}

impl GltfParser {
    pub fn dump<P: AsRef<Path>> (path: P) -> anyhow::Result<()> {
        let gltf = gltf::Gltf::open(path)?;
        for scene in gltf.scenes() {
            let scene_label = match scene.name() {
                Some(name) => name.to_owned(),
                None => scene.index().to_string()
            };
            println!("Scene {}", scene_label);

            for node in scene.nodes() {
                Self::dump_nodes(&node, 1);
            }
        }
        Ok(())
    }

    fn dump_nodes(node: &gltf::Node, depth: u32) {
        let node_label = Self::get_node_label(node);
        let indent_string = "  ".repeat(depth as usize);
        println!("{}Node {}", indent_string, node_label);
        for child in node.children() {
            Self::dump_nodes(&child, depth + 1);
        }
    }

    fn get_node_label(node: &gltf::Node) -> String {
        match node.name() {
            Some(name) => name.to_owned(),
            None => node.index().to_string()
        }
    }
}