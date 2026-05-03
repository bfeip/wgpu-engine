/// Opaque unique identifier used for all scene resources.
///
/// Backed by UUID v7 (time-ordered) for stable, globally-unique IDs that merge
/// cleanly across scenes and processes. The underlying representation is an
/// implementation detail — callers use `Id::new()` and `Id::nil()` only.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Id(uuid::Uuid);

impl Id {
    /// Creates a new globally-unique ID.
    pub fn new() -> Self {
        Self(uuid::Uuid::now_v7())
    }

    /// Returns the nil (all-zeros) ID. Useful as a sentinel or default value.
    pub fn nil() -> Self {
        Self(uuid::Uuid::nil())
    }

    /// Returns true if this is the nil ID.
    pub fn is_nil(&self) -> bool {
        self.0.is_nil()
    }
}

impl std::fmt::Debug for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Id({})", self.0)
    }
}

impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
