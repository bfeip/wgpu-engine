use std::io::{Read, Write};

use serde::{Deserialize, Serialize};

use duck_engine_scene::{Id, NodeId, EnvironmentMapId};

// ============================================================================
// Constants
// ============================================================================

/// Magic number identifying Duck files: "DUCK" in ASCII
pub const MAGIC: [u8; 4] = *b"DUCK";

/// Current format version (major.minor encoded as single u16)
/// major = version >> 8, minor = version & 0xFF
pub const VERSION: u16 = 0x0005; // 0.5 — Flat per-resource format; individual resource TOC entries

/// Size of the fixed header in bytes
pub const HEADER_SIZE: usize = std::mem::size_of::<FileHeader>();

// ============================================================================
// Resource Types
// ============================================================================

/// Identifies the type and encoding of a resource in the file.
///
/// The variant determines both what the resource is and how its bytes are encoded:
/// - `Texture`: raw PNG or JPEG bytes (already compressed; no outer wrapper)
/// - All others: zstd-compressed bincode of the corresponding scene type
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// Scene metadata (name, timestamps, generator info). resource_id = Id::nil().
    Metadata = 0,
    /// Scene graph node. resource_id = node.id.
    Node = 1,
    /// Mesh-material instance binding. resource_id = instance.id.
    Instance = 2,
    /// Material definition. resource_id = material.id.
    Material = 3,
    /// Mesh geometry data. resource_id = mesh.id.
    Mesh = 4,
    /// Embedded texture image. resource_id = texture.id. Bytes are raw PNG/JPEG.
    Texture = 5,
    /// Environment map. resource_id = environment_map.id.
    EnvironmentMap = 6,
}

// ============================================================================
// File Header & TOC
// ============================================================================

/// Fixed-size file header at the start of every .duck file.
#[derive(Debug, Clone)]
pub struct FileHeader {
    /// Magic number (must be MAGIC)
    pub magic: [u8; 4],
    /// Format version
    pub version: u16,
    /// Reserved flags
    pub flags: u16,
    /// Byte offset to the table of contents
    pub toc_offset: u64,
}

impl FileHeader {
    pub fn new(toc_offset: u64) -> Self {
        Self { magic: MAGIC, version: VERSION, flags: 0, toc_offset }
    }

    pub fn write<W: Write>(&self, writer: &mut W) -> Result<(), super::FormatError> {
        writer.write_all(&self.magic)?;
        writer.write_all(&self.version.to_le_bytes())?;
        writer.write_all(&self.flags.to_le_bytes())?;
        writer.write_all(&self.toc_offset.to_le_bytes())?;
        Ok(())
    }

    pub fn read<R: Read>(reader: &mut R) -> Result<Self, super::FormatError> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;

        if magic != MAGIC {
            return Err(super::FormatError::InvalidMagic);
        }

        let mut version_bytes = [0u8; 2];
        reader.read_exact(&mut version_bytes)?;
        let version = u16::from_le_bytes(version_bytes);

        let major = (version >> 8) as u8;
        let minor = (version & 0xFF) as u8;
        if major != 0 || minor != 5 {
            return Err(super::FormatError::UnsupportedVersion(major, minor));
        }

        let mut flags_bytes = [0u8; 2];
        reader.read_exact(&mut flags_bytes)?;
        let flags = u16::from_le_bytes(flags_bytes);

        let mut toc_offset_bytes = [0u8; 8];
        reader.read_exact(&mut toc_offset_bytes)?;
        let toc_offset = u64::from_le_bytes(toc_offset_bytes);

        Ok(Self { magic, version, flags, toc_offset })
    }
}

/// Entry in the table of contents describing one resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TocEntry {
    /// Resource type (also determines encoding)
    pub resource_type: ResourceType,
    /// Stable resource ID. Matches the scene ID for typed resources; nil for Metadata.
    pub resource_id: Id,
    /// Byte offset from start of file to the first byte of this resource's data.
    pub offset: u64,
    /// Size of the resource data in bytes.
    pub size: u32,
}

// ============================================================================
// Serializable Data Structures
// ============================================================================

/// Scene metadata stored in every .duck file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedMetadata {
    /// Optional scene name
    pub name: Option<String>,
    /// Unix timestamp of creation
    pub created_at: u64,
    /// Generator string (e.g., "duck-engine 0.1.0")
    pub generator: String,
    /// Root node IDs
    pub root_nodes: Vec<NodeId>,
    /// Active environment map ID, if any
    pub active_environment_map: Option<EnvironmentMapId>,
}
