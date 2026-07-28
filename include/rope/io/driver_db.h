#pragma once
// Space-weather database (O(log N) lookup) and hourly driver-window builder.

#include "rope/core/datetime.h"
#include "rope/io/csv_reader.h"

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rope::io {

// One hour of driver data. `t1`-`t4` are harmonic features always derived
// from `tp`, computed unconditionally whenever a row is built. `raw` holds
// whatever other named columns this row's source (CSV or .swbin) actually
// provided — commonly `f10`/`kp`, but not limited to any fixed set; a
// model's driver set is defined entirely by its manifest, not by this type.
struct DriverRow {
    TimePoint tp;
    float t1, t2, t3, t4;  // sin/cos harmonic features, always derived from tp

    std::vector<std::pair<std::string, float>> raw;

    // Resolves `name`: t1-t4 always come from the fields above (a raw column
    // with one of these names is rejected at load time, never silently
    // shadowed); "doy"/"hour_int" fall back to a value derived from `tp` when
    // not present in `raw`; anything else must be present in `raw`. Throws
    // std::runtime_error if `name` can't be resolved — never substitutes.
    float get(std::string_view name) const;
};

// The small, fixed vocabulary of driver names the framework can always
// resolve without reading them from a raw data source — t1-t4 (always) and
// doy/hour_int (fallback, only when absent from the raw source). Kept in
// sync with rope-registry's driver_registry.json "derived" entries by
// tests/cpp/test_driver_registry.cpp.
inline std::vector<std::string> known_derived_driver_names() {
    return {"t1", "t2", "t3", "t4", "doy", "hour_int"};
}

class SpaceWeatherBin;  // forward-declare for friend

class SpaceWeatherDB {
public:
    // Format auto-detected from extension (".swbin" -> binary, else CSV).
    static SpaceWeatherDB from_file(const std::filesystem::path& path);

    explicit SpaceWeatherDB(const std::filesystem::path& csv_path);

    // Look up a single TimePoint; throws if not found.
    DriverRow lookup(TimePoint tp) const;

    TimePoint time_min() const { return times_.front(); }
    TimePoint time_max() const { return times_.back(); }
    std::size_t size()   const { return times_.size(); }

private:
    friend class SpaceWeatherBin;

    // Private constructor used by SpaceWeatherBin::load().
    SpaceWeatherDB(std::vector<TimePoint>          times,
                   std::vector<std::string>        raw_names,
                   std::vector<std::vector<float>> raw_data);

    // Throws if `names` contains a reserved always-derived name (t1-t4) —
    // those may never be supplied by a raw source, CSV or binary alike.
    static void reject_reserved_names(const std::vector<std::string>& names);

    std::vector<TimePoint>          times_;
    std::vector<std::string>        raw_names_;
    std::vector<std::vector<float>> raw_data_;  // raw_data_[j][i] = raw_names_[j]'s value at row i

    DriverRow make_row(std::size_t idx) const;
};

class DriverWindowBuilder {
public:
    // Returns (seq_len-1 + horizon) DriverRows in chronological order; throws if any hourly slot is missing.
    static std::vector<DriverRow> build(const SpaceWeatherDB& db,
                                        std::string_view      start_iso,
                                        int horizon,
                                        int seq_len);
};

} // namespace rope::io
