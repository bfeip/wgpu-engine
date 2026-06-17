use std::sync::Mutex;

use crate::operator::Operator;

mod app;
mod context;
mod device;
mod dispatcher;

pub use app::AppEvent;
pub use context::EventContext;
pub use device::{DeviceEvent, DeviceEventKind};
pub use dispatcher::EventDispatcher;

/// Bridges `Arc<Mutex<T>>` into the dispatcher's type-erased storage.
///
/// Implemented blanket-wise for `Mutex<T: Operator>`, allowing `Arc<Mutex<T>>` to coerce
/// to `Arc<dyn ArcOperator>`. The `as_ptr` method returns the allocation address of the
/// `Mutex<T>` for pointer-equality identity checks.
pub(crate) trait ArcOperator {
    fn dispatch(&self, event: &Event, ctx: &mut EventContext) -> bool;
    fn name(&self) -> String;
    fn as_ptr(&self) -> *const ();
}

impl<T: Operator> ArcOperator for Mutex<T> {
    fn dispatch(&self, event: &Event, ctx: &mut EventContext) -> bool {
        self.lock().unwrap().dispatch(event, ctx)
    }
    fn name(&self) -> String {
        self.lock().unwrap().name().to_string()
    }
    fn as_ptr(&self) -> *const () {
        self as *const Mutex<T> as *const ()
    }
}

/// A single event flowing through the [`EventDispatcher`].
pub enum Event {
    /// A low-level input/device event.
    Device(DeviceEvent),
    /// A high-level semantic event emitted by an operator.
    App(AppEvent),
}

impl From<DeviceEvent> for Event {
    fn from(e: DeviceEvent) -> Self {
        Event::Device(e)
    }
}

impl From<AppEvent> for Event {
    fn from(e: AppEvent) -> Self {
        Event::App(e)
    }
}
