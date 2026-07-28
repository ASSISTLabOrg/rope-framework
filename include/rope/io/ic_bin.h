#pragma once
// Binary IC-table format (.icbin).
//
// Header (16 bytes, little-endian):
//   uint32  magic      = 0x52504943  ("RPIC")
//   uint32  version    = 2
//   uint32  nrows
//   uint32  latent_dim  (K)
//
// Axis name table (exactly 2 entries, immediately after the header):
//   uint32  name_len
//   char    name[name_len]           -- not NUL-terminated
//
// Records (nrows × (2 + K) × 4 bytes, little-endian):
//   float32 axis0_value
//   float32 axis1_value
//   float32 y[K]
//
// Axis identity is not fixed to F10/Kp — whatever 2 names the source table
// declares (see ICTable::axis_names()), cross-checked by the caller against
// the manifest's ic.params.grid_axes.

#include "rope/io/ic_table.h"

#include <filesystem>

namespace rope::io {

class IcBin {
public:
    // Load a .icbin file and return an ICTable ready for interpolation.
    // Throws on bad magic, unsupported version, or read failure.
    static ICTable load(const std::filesystem::path& bin_path);

    // Serialise an existing ICTable to .icbin.
    static void save(const ICTable& table,
                     const std::filesystem::path& bin_path);
};

} // namespace rope::io
