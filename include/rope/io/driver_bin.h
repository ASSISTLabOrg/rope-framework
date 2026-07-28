#pragma once
// Binary space-weather format (.swbin).
//
// Header (16 bytes, little-endian):
//   uint32  magic    = 0x52505357  ("RPSW")
//   uint32  version  = 2
//   uint32  nrows
//   uint32  ncols                      -- count of raw driver columns
//
// Name table (ncols entries, immediately after the header):
//   uint32  name_len
//   char    name[name_len]             -- not NUL-terminated
//
// Records (nrows × (8 + 4*ncols) bytes, little-endian):
//   int64   tp                          -- Unix timestamp (seconds since epoch)
//   float32 col[0..ncols)                -- one value per name-table entry, in order
//
// The raw column set is whatever the source that produced this file provides
// (commonly f10/kp, but not limited to them) — never a fixed schema. Derived
// features (t1-t4, and doy/hour_int when absent from the raw set) are never
// stored here; see DriverRow::get().

#include "rope/io/driver_db.h"

#include <filesystem>

namespace rope::io {

class SpaceWeatherBin {
public:
    // Load a .swbin file and return a SpaceWeatherDB ready for lookup.
    // Throws on bad magic, unsupported version, or read failure.
    static SpaceWeatherDB load(const std::filesystem::path& bin_path);

    // Serialise an existing SpaceWeatherDB to .swbin.
    static void save(const SpaceWeatherDB& db,
                     const std::filesystem::path& bin_path);
};

} // namespace rope::io
