mod navigation;

pub use navigation::NavigationOperator;

use crate::event::{CallbackId, EventDispatcher};

/// Unique identifier for an operator.
pub type OperatorId = u32;

/// Priority value for an operator. Lower values mean higher priority.
pub type OperatorPriority = u32;

/// Identifiers for known operators
pub enum BuiltinOperatorId {
    Navigation = 0
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
