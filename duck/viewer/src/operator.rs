mod navigation;
mod selection;
mod transform;

pub use navigation::{NavigationMode, NavigationOperator};
pub use selection::SelectionOperator;
pub use transform::TransformOperator;

use std::any::TypeId;

use crate::event::{CallbackId, EventDispatcher};

/// Unique identifier for an operator, derived from its concrete type via [`std::any::TypeId`].
pub type OperatorId = TypeId;

/// Operators encapsulate interaction logic and manage their own state. They register
/// callbacks with the EventDispatcher and can be dynamically activated/deactivated.
///
/// Each operator is responsible for tracking its own callback IDs and cleaning them up.
/// Ordering is managed by the OperatorManager - operators earlier in the stack receive
/// events first.
///
/// The `'static` bound means operators cannot hold borrowed references; use `Rc` or `Arc`
/// for shared state.
pub trait Operator: 'static {
    /// Called when the operator is activated.
    ///
    /// The operator should register its callbacks with the dispatcher and store
    /// the returned callback IDs for later cleanup.
    fn activate(&mut self, dispatcher: &mut EventDispatcher);

    /// Called when the operator is deactivated.
    ///
    /// The operator should unregister all its callbacks from the dispatcher.
    fn deactivate(&mut self, dispatcher: &mut EventDispatcher);

    /// Returns the unique ID for this operator, derived from its concrete type.
    fn id(&self) -> OperatorId {
        TypeId::of::<Self>()
    }

    /// Get a human-readable name for this operator.
    fn name(&self) -> &str;

    /// Get the callback IDs that this operator has registered.
    ///
    /// Returns an empty slice if the operator is not active.
    fn callback_ids(&self) -> &[CallbackId];

    /// Check if this operator is currently active, and thus has registered callbacks.
    fn is_active(&self) -> bool;
}

/// Manages a collection of active operators and their lifecycle.
///
/// The OperatorManager maintains operators in a stack-like structure where
/// operators at the front of the stack receive events first. Operators can
/// be added to the front or back, and reordered dynamically.
pub struct OperatorManager {
    operators: Vec<Box<dyn Operator>>,
}

impl OperatorManager {
    /// Creates a new empty operator manager.
    pub(crate) fn new() -> Self {
        Self {
            operators: Vec::new(),
        }
    }

    /// Adds an operator to the front of the stack (highest priority).
    ///
    /// The operator will receive events before all other operators.
    /// The operator's `activate` method is called to register its callbacks.
    pub fn push_front(
        &mut self,
        mut operator: Box<dyn Operator>,
        dispatcher: &mut EventDispatcher,
    ) {
        if !operator.is_active() {
            operator.activate(dispatcher);
        }
        self.operators.insert(0, operator);
        self.reorder_callbacks(dispatcher);
    }

    /// Adds an operator to the back of the stack (lowest priority).
    ///
    /// The operator will receive events after all other operators.
    /// The operator's `activate` method is called to register its callbacks.
    pub fn push_back(
        &mut self,
        mut operator: Box<dyn Operator>,
        dispatcher: &mut EventDispatcher,
    ) {
        if !operator.is_active() {
            operator.activate(dispatcher);
        }
        self.operators.push(operator);
        self.reorder_callbacks(dispatcher);
    }

    /// Removes an operator by ID and deactivates it.
    ///
    /// Returns `true` if an operator with that ID was found and removed.
    pub fn remove(&mut self, id: OperatorId, dispatcher: &mut EventDispatcher) -> bool {
        if let Some(pos) = self.operators.iter().position(|op| op.id() == id) {
            let mut operator = self.operators.remove(pos);
            if operator.is_active() {
                operator.deactivate(dispatcher);
            }
            self.reorder_callbacks(dispatcher);
            true
        } else {
            false
        }
    }

    /// Moves an operator to the front of the stack (highest priority).
    ///
    /// Returns `true` if the operator was found and moved.
    pub fn move_to_front(&mut self, id: OperatorId, dispatcher: &mut EventDispatcher) -> bool {
        if let Some(pos) = self.operators.iter().position(|op| op.id() == id) {
            if pos > 0 {
                let operator = self.operators.remove(pos);
                self.operators.insert(0, operator);
                self.reorder_callbacks(dispatcher);
            }
            true
        } else {
            false
        }
    }

    /// Moves an operator to the back of the stack (lowest priority).
    ///
    /// Returns `true` if the operator was found and moved.
    pub fn move_to_back(&mut self, id: OperatorId, dispatcher: &mut EventDispatcher) -> bool {
        if let Some(pos) = self.operators.iter().position(|op| op.id() == id) {
            if pos < self.operators.len() - 1 {
                let operator = self.operators.remove(pos);
                self.operators.push(operator);
                self.reorder_callbacks(dispatcher);
            }
            true
        } else {
            false
        }
    }

    /// Swaps the positions of two operators by their IDs.
    ///
    /// Returns `true` if both operators were found and swapped.
    /// If either operator is not found, no change is made and returns `false`.
    pub fn swap(&mut self, id1: OperatorId, id2: OperatorId, dispatcher: &mut EventDispatcher) -> bool {
        let pos1 = self.operators.iter().position(|op| op.id() == id1);
        let pos2 = self.operators.iter().position(|op| op.id() == id2);

        match (pos1, pos2) {
            (Some(p1), Some(p2)) if p1 != p2 => {
                self.operators.swap(p1, p2);
                self.reorder_callbacks(dispatcher);
                true
            }
            (Some(_), Some(_)) => true, // Same position (same ID), no-op but success
            _ => false,
        }
    }

    /// Reorders all callbacks in the EventDispatcher to match operator order.
    ///
    /// This ensures that when multiple operators register callbacks for the same
    /// event kind, they are invoked in the correct order.
    fn reorder_callbacks(&self, dispatcher: &mut EventDispatcher) {
        let ordered_ids: Vec<CallbackId> = self
            .operators
            .iter()
            .flat_map(|op| op.callback_ids().iter().copied())
            .collect();
        dispatcher.reorder(&ordered_ids);
    }

    /// Returns the number of operators.
    pub fn len(&self) -> usize {
        self.operators.len()
    }

    /// Returns `true` if the manager is empty
    pub fn is_empty(&self) -> bool {
        self.operators.is_empty()
    }

    /// Returns an iterator over the operators in priority order (front to back).
    pub fn iter(&self) -> impl Iterator<Item = &dyn Operator> {
        self.operators.iter().map(|op| op.as_ref())
    }

    /// Returns the operator at the front of the stack (highest priority), if any.
    pub fn front(&self) -> Option<&dyn Operator> {
        self.operators.first().map(|op| op.as_ref())
    }

    /// Returns the ID of the operator at the front of the stack, if any.
    pub fn front_id(&self) -> Option<OperatorId> {
        self.operators.first().map(|op| op.id())
    }

    /// Returns the position of an operator in the stack by ID.
    ///
    /// Position 0 is the front (highest priority).
    /// Returns `None` if the operator is not found.
    pub fn position(&self, id: OperatorId) -> Option<usize> {
        self.operators.iter().position(|op| op.id() == id)
    }

    /// Removes the operator of type `T` from the stack.
    pub fn remove_typed<T: Operator + 'static>(&mut self, dispatcher: &mut EventDispatcher) -> bool {
        self.remove(TypeId::of::<T>(), dispatcher)
    }

    /// Moves the operator of type `T` to the front of the stack (highest priority).
    pub fn move_to_front_typed<T: Operator + 'static>(&mut self, dispatcher: &mut EventDispatcher) -> bool {
        self.move_to_front(TypeId::of::<T>(), dispatcher)
    }

    /// Moves the operator of type `T` to the back of the stack (lowest priority).
    pub fn move_to_back_typed<T: Operator + 'static>(&mut self, dispatcher: &mut EventDispatcher) -> bool {
        self.move_to_back(TypeId::of::<T>(), dispatcher)
    }

    /// Swaps the positions of two operators by type.
    pub fn swap_typed<T: Operator + 'static, U: Operator + 'static>(&mut self, dispatcher: &mut EventDispatcher) -> bool {
        self.swap(TypeId::of::<T>(), TypeId::of::<U>(), dispatcher)
    }

    /// Returns the position of the operator of type `T` in the stack.
    pub fn position_typed<T: Operator + 'static>(&self) -> Option<usize> {
        self.position(TypeId::of::<T>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::TypeId;
    use std::cell::Cell;
    use std::rc::Rc;

    macro_rules! make_mock_op {
        ($name:ident, $cb1:expr, $cb2:expr) => {
            struct $name {
                active: bool,
                callbacks: Vec<CallbackId>,
                activate_count: Rc<Cell<u32>>,
                deactivate_count: Rc<Cell<u32>>,
            }

            impl $name {
                fn new(
                    activate_count: Rc<Cell<u32>>,
                    deactivate_count: Rc<Cell<u32>>,
                ) -> Self {
                    Self {
                        active: false,
                        callbacks: Vec::new(),
                        activate_count,
                        deactivate_count,
                    }
                }
            }

            impl Operator for $name {
                fn activate(&mut self, _dispatcher: &mut EventDispatcher) {
                    self.active = true;
                    self.callbacks.push($cb1);
                    self.callbacks.push($cb2);
                    self.activate_count.set(self.activate_count.get() + 1);
                }

                fn deactivate(&mut self, _dispatcher: &mut EventDispatcher) {
                    self.active = false;
                    self.callbacks.clear();
                    self.deactivate_count.set(self.deactivate_count.get() + 1);
                }

                fn name(&self) -> &str {
                    stringify!($name)
                }

                fn callback_ids(&self) -> &[CallbackId] {
                    &self.callbacks
                }

                fn is_active(&self) -> bool {
                    self.active
                }
            }
        };
    }

    make_mock_op!(MockOp1, 101, 201);
    make_mock_op!(MockOp2, 102, 202);
    make_mock_op!(MockOp3, 103, 203);
    // Used only as a TypeId for "not found" tests — never instantiated.
    struct MockOpUnused;
    impl Operator for MockOpUnused {
        fn activate(&mut self, _: &mut EventDispatcher) {}
        fn deactivate(&mut self, _: &mut EventDispatcher) {}
        fn name(&self) -> &str { "MockOpUnused" }
        fn callback_ids(&self) -> &[CallbackId] { &[] }
        fn is_active(&self) -> bool { false }
    }

    fn counts() -> (Rc<Cell<u32>>, Rc<Cell<u32>>) {
        (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)))
    }

    // ===== OperatorManager Tests =====

    #[test]
    fn test_operator_manager_new() {
        let manager = OperatorManager::new();
        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);
    }

    #[test]
    fn test_push_front() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();
        let (ac, dc) = counts();

        let op1 = Box::new(MockOp1::new(Rc::clone(&ac), Rc::clone(&dc)));
        manager.push_front(op1, &mut dispatcher);
        assert_eq!(manager.len(), 1);

        // Op2 pushed to front should become first
        let op2 = Box::new(MockOp2::new(Rc::clone(&ac), Rc::clone(&dc)));
        manager.push_front(op2, &mut dispatcher);
        assert_eq!(manager.len(), 2);

        let names: Vec<&str> = manager.iter().map(|op| op.name()).collect();
        assert_eq!(names, vec!["MockOp2", "MockOp1"]);
    }

    #[test]
    fn test_push_back() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();
        let (ac, dc) = counts();

        let op1 = Box::new(MockOp1::new(Rc::clone(&ac), Rc::clone(&dc)));
        manager.push_back(op1, &mut dispatcher);

        let op2 = Box::new(MockOp2::new(Rc::clone(&ac), Rc::clone(&dc)));
        manager.push_back(op2, &mut dispatcher);

        let names: Vec<&str> = manager.iter().map(|op| op.name()).collect();
        assert_eq!(names, vec!["MockOp1", "MockOp2"]);
    }

    #[test]
    fn test_push_front_activates() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();
        let (ac, dc) = counts();

        let op = Box::new(MockOp1::new(Rc::clone(&ac), Rc::clone(&dc)));
        assert!(!op.is_active());

        manager.push_front(op, &mut dispatcher);

        assert_eq!(ac.get(), 1);
        assert_eq!(dc.get(), 0);

        let op_ref = manager.iter().next().unwrap();
        assert!(op_ref.is_active());
        assert_eq!(op_ref.callback_ids().len(), 2);
    }

    #[test]
    fn test_remove_deactivates() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();
        let (ac, dc) = counts();

        manager.push_front(Box::new(MockOp1::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);
        assert_eq!(ac.get(), 1);
        assert_eq!(dc.get(), 0);

        let removed = manager.remove(TypeId::of::<MockOp1>(), &mut dispatcher);
        assert!(removed);

        assert_eq!(ac.get(), 1);
        assert_eq!(dc.get(), 1);
        assert!(manager.is_empty());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();
        let (ac, dc) = counts();

        manager.push_front(Box::new(MockOp1::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);

        let removed = manager.remove(TypeId::of::<MockOpUnused>(), &mut dispatcher);
        assert!(!removed);
        assert_eq!(manager.len(), 1);
        assert_eq!(dc.get(), 0);
    }

    #[test]
    fn test_move_to_front() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();
        let (ac, dc) = counts();

        manager.push_back(Box::new(MockOp1::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);
        manager.push_back(Box::new(MockOp2::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);
        manager.push_back(Box::new(MockOp3::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);

        let moved = manager.move_to_front(TypeId::of::<MockOp3>(), &mut dispatcher);
        assert!(moved);

        let names: Vec<&str> = manager.iter().map(|op| op.name()).collect();
        assert_eq!(names, vec!["MockOp3", "MockOp1", "MockOp2"]);
    }

    #[test]
    fn test_move_to_back() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();
        let (ac, dc) = counts();

        manager.push_back(Box::new(MockOp1::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);
        manager.push_back(Box::new(MockOp2::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);
        manager.push_back(Box::new(MockOp3::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);

        let moved = manager.move_to_back(TypeId::of::<MockOp1>(), &mut dispatcher);
        assert!(moved);

        let names: Vec<&str> = manager.iter().map(|op| op.name()).collect();
        assert_eq!(names, vec!["MockOp2", "MockOp3", "MockOp1"]);
    }

    #[test]
    fn test_move_nonexistent() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();
        let (ac, dc) = counts();

        manager.push_back(Box::new(MockOp1::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);

        assert!(!manager.move_to_front(TypeId::of::<MockOpUnused>(), &mut dispatcher));
        assert!(!manager.move_to_back(TypeId::of::<MockOpUnused>(), &mut dispatcher));
    }

    #[test]
    fn test_swap() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();
        let (ac, dc) = counts();

        manager.push_back(Box::new(MockOp1::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);
        manager.push_back(Box::new(MockOp2::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);
        manager.push_back(Box::new(MockOp3::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);

        let swapped = manager.swap(TypeId::of::<MockOp1>(), TypeId::of::<MockOp3>(), &mut dispatcher);
        assert!(swapped);
        let names: Vec<&str> = manager.iter().map(|op| op.name()).collect();
        assert_eq!(names, vec!["MockOp3", "MockOp2", "MockOp1"]);

        let swapped = manager.swap(TypeId::of::<MockOp1>(), TypeId::of::<MockOp3>(), &mut dispatcher);
        assert!(swapped);
        let names: Vec<&str> = manager.iter().map(|op| op.name()).collect();
        assert_eq!(names, vec!["MockOp1", "MockOp2", "MockOp3"]);
    }

    #[test]
    fn test_swap_nonexistent() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();
        let (ac, dc) = counts();

        manager.push_back(Box::new(MockOp1::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);
        manager.push_back(Box::new(MockOp2::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);

        let swapped = manager.swap(TypeId::of::<MockOp1>(), TypeId::of::<MockOpUnused>(), &mut dispatcher);
        assert!(!swapped);

        let names: Vec<&str> = manager.iter().map(|op| op.name()).collect();
        assert_eq!(names, vec!["MockOp1", "MockOp2"]);
    }

    #[test]
    fn test_swap_same_id() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();
        let (ac, dc) = counts();

        manager.push_back(Box::new(MockOp1::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);

        let swapped = manager.swap(TypeId::of::<MockOp1>(), TypeId::of::<MockOp1>(), &mut dispatcher);
        assert!(swapped);

        let names: Vec<&str> = manager.iter().map(|op| op.name()).collect();
        assert_eq!(names, vec!["MockOp1"]);
    }

    #[test]
    fn test_iter_order() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();
        let (ac, dc) = counts();

        manager.push_back(Box::new(MockOp1::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);
        manager.push_back(Box::new(MockOp2::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);
        manager.push_back(Box::new(MockOp3::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);

        let mut iter = manager.iter();

        let first = iter.next().unwrap();
        assert_eq!(first.id(), TypeId::of::<MockOp1>());
        assert_eq!(first.name(), "MockOp1");

        let second = iter.next().unwrap();
        assert_eq!(second.id(), TypeId::of::<MockOp2>());
        assert_eq!(second.name(), "MockOp2");

        let third = iter.next().unwrap();
        assert_eq!(third.id(), TypeId::of::<MockOp3>());
        assert_eq!(third.name(), "MockOp3");

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_position() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();
        let (ac, dc) = counts();

        manager.push_back(Box::new(MockOp1::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);
        manager.push_back(Box::new(MockOp2::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);
        manager.push_back(Box::new(MockOp3::new(Rc::clone(&ac), Rc::clone(&dc))), &mut dispatcher);

        assert_eq!(manager.position(TypeId::of::<MockOp1>()), Some(0));
        assert_eq!(manager.position(TypeId::of::<MockOp2>()), Some(1));
        assert_eq!(manager.position(TypeId::of::<MockOp3>()), Some(2));
        assert_eq!(manager.position(TypeId::of::<MockOpUnused>()), None);

        manager.swap(TypeId::of::<MockOp1>(), TypeId::of::<MockOp3>(), &mut dispatcher);
        assert_eq!(manager.position(TypeId::of::<MockOp1>()), Some(2));
        assert_eq!(manager.position(TypeId::of::<MockOp3>()), Some(0));
    }
}
