use duck_engine_viewer::common::Point3;
use duck_engine_viewer::operator::Operator;

pub trait ModelingTool: Operator {
    /// Clean up in-progress state (preview nodes, hidden geometry).
    /// Called automatically before any tool switch.
    fn deactivate(&mut self);

    /// True when the tool completed or was cancelled and should cede back to selection.
    fn is_finished(&self) -> bool {
        false
    }

    /// The world-space point this tool wants the modeler's 3D cursor to mark
    /// (e.g. the current snap location), or `None` to hide it. Polled each frame
    /// while the tool is active.
    fn cursor_target(&self) -> Option<Point3> {
        None
    }
}
