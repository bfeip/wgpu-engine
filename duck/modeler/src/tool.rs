use duck_engine_viewer::operator::Operator;

pub trait ModelingTool: Operator {
    /// Clean up in-progress state (preview nodes, hidden geometry).
    /// Called automatically before any tool switch.
    fn deactivate(&mut self);

    /// True when the tool completed or was cancelled and should cede back to selection.
    fn is_finished(&self) -> bool {
        false
    }
}
