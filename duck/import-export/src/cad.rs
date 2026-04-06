//! CAD format support: STEP and IGES.
//!
//! Re-exports constants and helpers used by [`CadImporter`](crate::importer::CadImporter)
//! and exposed so callers (e.g. file dialogs) can enumerate supported extensions.

/// File extensions handled by the CAD importer.
pub const CAD_EXTENSIONS: &[&str] = &["step", "stp", "iges", "igs"];

/// Returns true if `ext` is a CAD file extension (case-insensitive).
pub fn is_cad_extension(ext: &str) -> bool {
    CAD_EXTENSIONS.iter().any(|e| e.eq_ignore_ascii_case(ext))
}

/// Returns true if `ext` identifies a STEP file specifically.
pub fn is_step_extension(ext: &str) -> bool {
    ext.eq_ignore_ascii_case("step") || ext.eq_ignore_ascii_case("stp")
}
