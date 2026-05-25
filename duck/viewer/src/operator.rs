mod navigation;
mod selection;
mod transform;

pub use navigation::{NavigationMode, NavigationOperator};
pub use selection::SelectionOperator;
pub use transform::TransformOperator;

use std::any::TypeId;

use crate::event::{Event, EventContext};

/// Unique identifier for an operator, derived from its concrete type via [`TypeId`].
pub type OperatorId = TypeId;

/// Operators encapsulate interaction logic and own their own state.
///
/// Each operator receives every dispatched event through [`dispatch`](Operator::dispatch)
/// and returns `true` to stop propagation or `false` to let subsequent operators handle it.
/// Operators are stored in the [`EventDispatcher`](crate::event::EventDispatcher) in priority
/// order; those inserted earlier receive events first.
pub trait Operator: 'static {
    /// Called for every dispatched event.
    ///
    /// Return `true` to consume the event and stop further propagation,
    /// or `false` to allow subsequent operators to handle it.
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool;

    /// Human-readable name for this operator (used for debugging and UI display).
    fn name(&self) -> &str;

    /// Unique identifier derived from the concrete type.
    ///
    /// The default implementation uses [`TypeId::of::<Self>()`], which means only
    /// one instance of each operator type can be active at a time.
    fn id(&self) -> OperatorId {
        TypeId::of::<Self>()
    }
}

impl dyn Operator {
    /// Attempts to downcast a shared reference to a concrete operator type.
    pub fn downcast_ref<T: Operator>(&self) -> Option<&T> {
        if self.id() == TypeId::of::<T>() {
            // SAFETY: id() confirmed the concrete type is T.
            Some(unsafe { &*(self as *const dyn Operator as *const T) })
        } else {
            None
        }
    }

    /// Attempts to downcast a mutable reference to a concrete operator type.
    pub fn downcast_mut<T: Operator>(&mut self) -> Option<&mut T> {
        if self.id() == TypeId::of::<T>() {
            // SAFETY: id() confirmed the concrete type is T.
            Some(unsafe { &mut *(self as *mut dyn Operator as *mut T) })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{Event, EventContext};

    struct MockOpA;
    impl Operator for MockOpA {
        fn dispatch(&mut self, _: &Event, _: &mut EventContext) -> bool { false }
        fn name(&self) -> &str { "MockOpA" }
    }

    struct MockOpB;
    impl Operator for MockOpB {
        fn dispatch(&mut self, _: &Event, _: &mut EventContext) -> bool { false }
        fn name(&self) -> &str { "MockOpB" }
    }

    #[test]
    fn downcast_ref_hit() {
        let op: Box<dyn Operator> = Box::new(MockOpA);
        assert!(op.downcast_ref::<MockOpA>().is_some());
    }

    #[test]
    fn downcast_ref_miss() {
        let op: Box<dyn Operator> = Box::new(MockOpA);
        assert!(op.downcast_ref::<MockOpB>().is_none());
    }

    #[test]
    fn downcast_mut_hit() {
        let mut op: Box<dyn Operator> = Box::new(MockOpA);
        assert!(op.downcast_mut::<MockOpA>().is_some());
    }

    #[test]
    fn downcast_mut_miss() {
        let mut op: Box<dyn Operator> = Box::new(MockOpA);
        assert!(op.downcast_mut::<MockOpB>().is_none());
    }
}
