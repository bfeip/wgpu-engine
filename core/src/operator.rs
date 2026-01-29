mod navigation;
mod selection;
mod transform;
mod walk;

pub use navigation::NavigationOperator;
pub use selection::SelectionOperator;
pub use transform::TransformOperator;
pub use walk::WalkOperator;

use crate::event::{CallbackId, EventDispatcher};

/// Unique identifier for an operator.
pub type OperatorId = u32;

/// Identifiers for known operators
pub enum BuiltinOperatorId {
    Navigation = 0,
    Selection = 1,
    Walk = 2,
    Transform = 3,
}

impl Into<OperatorId> for BuiltinOperatorId {
    fn into(self) -> OperatorId {
        self as OperatorId
    }
}

/// Operators encapsulate interaction logic and manage their own state. They register
/// callbacks with the EventDispatcher and can be dynamically activated/deactivated.
///
/// Each operator is responsible for tracking its own callback IDs and cleaning them up.
/// Ordering is managed by the OperatorManager - operators earlier in the stack receive
/// events first.
pub trait Operator {
    /// Called when the operator is activated.
    ///
    /// The operator should register its callbacks with the dispatcher and store
    /// the returned callback IDs for later cleanup.
    fn activate(&mut self, dispatcher: &mut EventDispatcher);

    /// Called when the operator is deactivated.
    ///
    /// The operator should unregister all its callbacks from the dispatcher.
    fn deactivate(&mut self, dispatcher: &mut EventDispatcher);

    /// Get the unique ID for this operator.
    fn id(&self) -> OperatorId;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    /// Mock operator for testing
    struct MockOperator {
        id: OperatorId,
        name: String,
        active: bool,
        callbacks: Vec<CallbackId>,
        activate_count: Rc<Cell<u32>>,
        deactivate_count: Rc<Cell<u32>>,
    }

    impl MockOperator {
        fn new(
            id: OperatorId,
            name: &str,
            activate_count: Rc<Cell<u32>>,
            deactivate_count: Rc<Cell<u32>>,
        ) -> Self {
            Self {
                id,
                name: name.to_string(),
                active: false,
                callbacks: Vec::new(),
                activate_count,
                deactivate_count,
            }
        }
    }

    impl Operator for MockOperator {
        fn activate(&mut self, _dispatcher: &mut EventDispatcher) {
            self.active = true;
            // Simulate registering 2 callbacks with unique IDs based on operator ID
            self.callbacks.push(100 + self.id);
            self.callbacks.push(200 + self.id);
            self.activate_count.set(self.activate_count.get() + 1);
        }

        fn deactivate(&mut self, _dispatcher: &mut EventDispatcher) {
            self.active = false;
            self.callbacks.clear();
            self.deactivate_count.set(self.deactivate_count.get() + 1);
        }

        fn id(&self) -> OperatorId {
            self.id
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn callback_ids(&self) -> &[CallbackId] {
            &self.callbacks
        }

        fn is_active(&self) -> bool {
            self.active
        }
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

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        // Add first operator
        let op1 = Box::new(MockOperator::new(1, "Op1", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_front(op1, &mut dispatcher);
        assert_eq!(manager.len(), 1);

        // Add second operator to front - should be first
        let op2 = Box::new(MockOperator::new(2, "Op2", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_front(op2, &mut dispatcher);
        assert_eq!(manager.len(), 2);

        // Op2 should be first
        let ids: Vec<OperatorId> = manager.iter().map(|op| op.id()).collect();
        assert_eq!(ids, vec![2, 1]);
    }

    #[test]
    fn test_push_back() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        // Add first operator
        let op1 = Box::new(MockOperator::new(1, "Op1", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op1, &mut dispatcher);

        // Add second operator to back - should be last
        let op2 = Box::new(MockOperator::new(2, "Op2", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op2, &mut dispatcher);

        // Op1 should still be first
        let ids: Vec<OperatorId> = manager.iter().map(|op| op.id()).collect();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn test_push_front_activates() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        let op = Box::new(MockOperator::new(1, "TestOp", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        assert!(!op.is_active());

        manager.push_front(op, &mut dispatcher);

        assert_eq!(activate_count.get(), 1);
        assert_eq!(deactivate_count.get(), 0);

        let op_ref = manager.iter().next().unwrap();
        assert!(op_ref.is_active());
        assert_eq!(op_ref.callback_ids().len(), 2);
    }

    #[test]
    fn test_remove_deactivates() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        let op = Box::new(MockOperator::new(1, "TestOp", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_front(op, &mut dispatcher);

        assert_eq!(activate_count.get(), 1);
        assert_eq!(deactivate_count.get(), 0);

        let removed = manager.remove(1, &mut dispatcher);
        assert!(removed);

        assert_eq!(activate_count.get(), 1);
        assert_eq!(deactivate_count.get(), 1);
        assert!(manager.is_empty());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        let op = Box::new(MockOperator::new(1, "Op1", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_front(op, &mut dispatcher);

        let removed = manager.remove(999, &mut dispatcher);
        assert!(!removed);
        assert_eq!(manager.len(), 1);
        assert_eq!(deactivate_count.get(), 0);
    }

    #[test]
    fn test_move_to_front() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        // Add three operators: 1, 2, 3 (in order)
        let op1 = Box::new(MockOperator::new(1, "Op1", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op1, &mut dispatcher);
        let op2 = Box::new(MockOperator::new(2, "Op2", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op2, &mut dispatcher);
        let op3 = Box::new(MockOperator::new(3, "Op3", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op3, &mut dispatcher);

        // Move op3 to front
        let moved = manager.move_to_front(3, &mut dispatcher);
        assert!(moved);

        let ids: Vec<OperatorId> = manager.iter().map(|op| op.id()).collect();
        assert_eq!(ids, vec![3, 1, 2]);
    }

    #[test]
    fn test_move_to_back() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        // Add three operators: 1, 2, 3 (in order)
        let op1 = Box::new(MockOperator::new(1, "Op1", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op1, &mut dispatcher);
        let op2 = Box::new(MockOperator::new(2, "Op2", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op2, &mut dispatcher);
        let op3 = Box::new(MockOperator::new(3, "Op3", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op3, &mut dispatcher);

        // Move op1 to back
        let moved = manager.move_to_back(1, &mut dispatcher);
        assert!(moved);

        let ids: Vec<OperatorId> = manager.iter().map(|op| op.id()).collect();
        assert_eq!(ids, vec![2, 3, 1]);
    }

    #[test]
    fn test_move_nonexistent() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        let op = Box::new(MockOperator::new(1, "Op1", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op, &mut dispatcher);

        let moved = manager.move_to_front(999, &mut dispatcher);
        assert!(!moved);

        let moved = manager.move_to_back(999, &mut dispatcher);
        assert!(!moved);
    }

    #[test]
    fn test_swap() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        // Add three operators: 1, 2, 3 (in order)
        let op1 = Box::new(MockOperator::new(1, "Op1", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op1, &mut dispatcher);
        let op2 = Box::new(MockOperator::new(2, "Op2", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op2, &mut dispatcher);
        let op3 = Box::new(MockOperator::new(3, "Op3", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op3, &mut dispatcher);

        // Swap op1 and op3
        let swapped = manager.swap(1, 3, &mut dispatcher);
        assert!(swapped);

        let ids: Vec<OperatorId> = manager.iter().map(|op| op.id()).collect();
        assert_eq!(ids, vec![3, 2, 1]);

        // Swap back
        let swapped = manager.swap(1, 3, &mut dispatcher);
        assert!(swapped);

        let ids: Vec<OperatorId> = manager.iter().map(|op| op.id()).collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_swap_nonexistent() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        let op1 = Box::new(MockOperator::new(1, "Op1", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op1, &mut dispatcher);
        let op2 = Box::new(MockOperator::new(2, "Op2", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op2, &mut dispatcher);

        // Swap with nonexistent operator
        let swapped = manager.swap(1, 999, &mut dispatcher);
        assert!(!swapped);

        // Order unchanged
        let ids: Vec<OperatorId> = manager.iter().map(|op| op.id()).collect();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn test_swap_same_id() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        let op1 = Box::new(MockOperator::new(1, "Op1", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op1, &mut dispatcher);

        // Swap with itself - should succeed but be a no-op
        let swapped = manager.swap(1, 1, &mut dispatcher);
        assert!(swapped);

        let ids: Vec<OperatorId> = manager.iter().map(|op| op.id()).collect();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn test_iter_order() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        // Add operators in specific order using push_front
        let op1 = Box::new(MockOperator::new(1, "First", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op1, &mut dispatcher);

        let op2 = Box::new(MockOperator::new(2, "Second", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op2, &mut dispatcher);

        let op3 = Box::new(MockOperator::new(3, "Third", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op3, &mut dispatcher);

        let mut iter = manager.iter();

        let first = iter.next().unwrap();
        assert_eq!(first.id(), 1);
        assert_eq!(first.name(), "First");

        let second = iter.next().unwrap();
        assert_eq!(second.id(), 2);
        assert_eq!(second.name(), "Second");

        let third = iter.next().unwrap();
        assert_eq!(third.id(), 3);
        assert_eq!(third.name(), "Third");

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_position() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        let op1 = Box::new(MockOperator::new(1, "Op1", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op1, &mut dispatcher);
        let op2 = Box::new(MockOperator::new(2, "Op2", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op2, &mut dispatcher);
        let op3 = Box::new(MockOperator::new(3, "Op3", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.push_back(op3, &mut dispatcher);

        assert_eq!(manager.position(1), Some(0));
        assert_eq!(manager.position(2), Some(1));
        assert_eq!(manager.position(3), Some(2));
        assert_eq!(manager.position(999), None);

        // After swap, positions change
        manager.swap(1, 3, &mut dispatcher);
        assert_eq!(manager.position(1), Some(2));
        assert_eq!(manager.position(3), Some(0));
    }
}
