mod navigation;
mod selection;

pub use navigation::NavigationOperator;
pub use selection::SelectionOperator;

use crate::event::{CallbackId, EventDispatcher};

/// Unique identifier for an operator.
pub type OperatorId = u32;

/// Priority value for an operator. Lower values mean higher priority.
pub type OperatorPriority = u32;

/// Identifiers for known operators
pub enum BuiltinOperatorId {
    Navigation = 0,
    Selection = 1,
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
/// Priority is managed by the OperatorManager.
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
/// The OperatorManager maintains operators in priority order and handles
/// activation/deactivation through the EventDispatcher.
/// Lower priority values mean that operator's callbacks are invoked first.
pub struct OperatorManager {
    operators: Vec<(OperatorPriority, Box<dyn Operator>)>,
}

impl OperatorManager {
    /// Creates a new empty operator manager.
    pub fn new() -> Self {
        Self {
            operators: Vec::new(),
        }
    }

    /// Adds an operator with the specified priority and activates it.
    ///
    /// The operator is inserted in priority order (highest priority first).
    /// The operator's `activate` method is called to register its callbacks
    /// (if not already active).
    ///
    /// Lower priority values mean callbacks are invoked first.
    pub fn add_operator(
        &mut self,
        mut operator: Box<dyn Operator>,
        priority: OperatorPriority,
        dispatcher: &mut EventDispatcher,
    ) {
        // Activate the operator if it's not already active
        if !operator.is_active() {
            operator.activate(dispatcher);
        }

        // Find the insertion position to maintain priority order (lowest value = highest priority)
        let insert_pos = self
            .operators
            .iter()
            .position(|(pri, _)| *pri > priority)
            .unwrap_or(self.operators.len());

        self.operators.insert(insert_pos, (priority, operator));

        // Reorder all callbacks in the dispatcher to match operator priority order
        self.reorder_callbacks(dispatcher);
    }

    /// Removes an operator by ID and deactivates it.
    ///
    /// Returns `true` if an operator with that ID was found and removed.
    pub fn remove_operator(&mut self, id: OperatorId, dispatcher: &mut EventDispatcher) -> bool {
        if let Some(pos) = self.operators.iter().position(|(_, op)| op.id() == id) {
            let (_, mut operator) = self.operators.remove(pos);
            if operator.is_active() {
                operator.deactivate(dispatcher);
            }
            self.reorder_callbacks(dispatcher);
            true
        } else {
            false
        }
    }

    /// Reorders all callbacks in the EventDispatcher to match operator priority order.
    ///
    /// This ensures that when multiple operators register callbacks for the same
    /// event kind, they are invoked in the correct priority order.
    fn reorder_callbacks(&self, dispatcher: &mut EventDispatcher) {
        // Collect all callback IDs in priority order (operators are already sorted)
        let ordered_ids: Vec<CallbackId> = self
            .operators
            .iter()
            .flat_map(|(_, op)| op.callback_ids().iter().copied())
            .collect();

        // Reorder all event kinds to match this priority order
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

    /// Returns an iterator over the active operators (ordered by priority).
    pub fn iter(&self) -> impl Iterator<Item = &dyn Operator> {
        self.operators.iter().map(|(_, op)| op.as_ref())
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
    fn test_operator_manager_len() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        assert_eq!(manager.len(), 0);

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));
        let op1 = Box::new(MockOperator::new(1, "Op1", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.add_operator(op1, 1, &mut dispatcher);

        assert_eq!(manager.len(), 1);

        let op2 = Box::new(MockOperator::new(2, "Op2", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.add_operator(op2, 2, &mut dispatcher);

        assert_eq!(manager.len(), 2);
    }

    #[test]
    fn test_operator_manager_is_empty() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        assert!(manager.is_empty());

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));
        let op = Box::new(MockOperator::new(1, "Op1", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.add_operator(op, 1, &mut dispatcher);

        assert!(!manager.is_empty());

        manager.remove_operator(1, &mut dispatcher);

        assert!(manager.is_empty());
    }

    #[test]
    fn test_add_operator_activates() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        let op = Box::new(MockOperator::new(
            1,
            "TestOp",
            Rc::clone(&activate_count),
            Rc::clone(&deactivate_count),
        ));

        // Operator should not be active initially
        assert!(!op.is_active());

        manager.add_operator(op, 1, &mut dispatcher);

        // Activate should have been called
        assert_eq!(activate_count.get(), 1);
        assert_eq!(deactivate_count.get(), 0);

        // Operator should now be active
        let op_ref = manager.iter().next().unwrap();
        assert!(op_ref.is_active());
        assert_eq!(op_ref.callback_ids().len(), 2); // MockOperator registers 2 callbacks
    }

    #[test]
    fn test_add_operator_priority_ordering() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        // Add operator with priority 10 (lower priority)
        let op1 = Box::new(MockOperator::new(1, "Op1", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.add_operator(op1, 10, &mut dispatcher);

        // Add operator with priority 5 (higher priority - should come first)
        let op2 = Box::new(MockOperator::new(2, "Op2", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.add_operator(op2, 5, &mut dispatcher);

        // Collect operator IDs in order
        let ids: Vec<OperatorId> = manager.iter().map(|op| op.id()).collect();

        // Op2 (priority 5) should come before Op1 (priority 10)
        assert_eq!(ids, vec![2, 1]);
    }

    #[test]
    fn test_add_operator_inserts_in_order() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        // Add operators with various priorities
        let op1 = Box::new(MockOperator::new(1, "Op1", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.add_operator(op1, 30, &mut dispatcher);

        let op2 = Box::new(MockOperator::new(2, "Op2", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.add_operator(op2, 10, &mut dispatcher);

        let op3 = Box::new(MockOperator::new(3, "Op3", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.add_operator(op3, 20, &mut dispatcher);

        let op4 = Box::new(MockOperator::new(4, "Op4", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.add_operator(op4, 5, &mut dispatcher);

        // Should be ordered by priority: 5, 10, 20, 30
        let ids: Vec<OperatorId> = manager.iter().map(|op| op.id()).collect();
        assert_eq!(ids, vec![4, 2, 3, 1]);

        // Verify priorities are correct
        let priorities: Vec<u32> = manager.operators.iter().map(|(pri, _)| *pri).collect();
        assert_eq!(priorities, vec![5, 10, 20, 30]);
    }

    #[test]
    fn test_remove_operator_deactivates() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        let op = Box::new(MockOperator::new(
            1,
            "TestOp",
            Rc::clone(&activate_count),
            Rc::clone(&deactivate_count),
        ));

        manager.add_operator(op, 1, &mut dispatcher);

        // Activate should have been called once
        assert_eq!(activate_count.get(), 1);
        assert_eq!(deactivate_count.get(), 0);

        // Remove the operator
        let removed = manager.remove_operator(1, &mut dispatcher);
        assert!(removed);

        // Deactivate should have been called
        assert_eq!(activate_count.get(), 1);
        assert_eq!(deactivate_count.get(), 1);

        // Manager should now be empty
        assert!(manager.is_empty());
    }

    #[test]
    fn test_remove_operator_reorders() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        // Add three operators
        let op1 = Box::new(MockOperator::new(1, "Op1", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.add_operator(op1, 10, &mut dispatcher);

        let op2 = Box::new(MockOperator::new(2, "Op2", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.add_operator(op2, 20, &mut dispatcher);

        let op3 = Box::new(MockOperator::new(3, "Op3", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.add_operator(op3, 30, &mut dispatcher);

        // Remove the middle operator
        let removed = manager.remove_operator(2, &mut dispatcher);
        assert!(removed);

        // Should have 2 operators left
        assert_eq!(manager.len(), 2);

        // Remaining operators should be in order
        let ids: Vec<OperatorId> = manager.iter().map(|op| op.id()).collect();
        assert_eq!(ids, vec![1, 3]);

        // Deactivate should have been called once (for op2)
        assert_eq!(deactivate_count.get(), 1);
    }

    #[test]
    fn test_remove_operator_nonexistent() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        let op = Box::new(MockOperator::new(1, "Op1", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.add_operator(op, 1, &mut dispatcher);

        // Try to remove an operator that doesn't exist
        let removed = manager.remove_operator(999, &mut dispatcher);
        assert!(!removed);

        // Manager should still have 1 operator
        assert_eq!(manager.len(), 1);

        // Deactivate should not have been called
        assert_eq!(deactivate_count.get(), 0);
    }

    #[test]
    fn test_operator_manager_iter() {
        let mut manager = OperatorManager::new();
        let mut dispatcher = EventDispatcher::new();

        let activate_count = Rc::new(Cell::new(0));
        let deactivate_count = Rc::new(Cell::new(0));

        // Add operators with specific priorities and names
        let op1 = Box::new(MockOperator::new(1, "LowPriority", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.add_operator(op1, 100, &mut dispatcher);

        let op2 = Box::new(MockOperator::new(2, "HighPriority", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.add_operator(op2, 10, &mut dispatcher);

        let op3 = Box::new(MockOperator::new(3, "MediumPriority", Rc::clone(&activate_count), Rc::clone(&deactivate_count)));
        manager.add_operator(op3, 50, &mut dispatcher);

        // Iterate and verify order (should be sorted by priority: 10, 50, 100)
        let mut iter = manager.iter();

        let first = iter.next().unwrap();
        assert_eq!(first.id(), 2);
        assert_eq!(first.name(), "HighPriority");
        assert!(first.is_active());

        let second = iter.next().unwrap();
        assert_eq!(second.id(), 3);
        assert_eq!(second.name(), "MediumPriority");
        assert!(second.is_active());

        let third = iter.next().unwrap();
        assert_eq!(third.id(), 1);
        assert_eq!(third.name(), "LowPriority");
        assert!(third.is_active());

        assert!(iter.next().is_none());
    }
}
