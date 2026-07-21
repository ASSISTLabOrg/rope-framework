#pragma once
// Space-weather database (O(log N) lookup) and hourly driver-window builder.

#include "rope/core/datetime.h"
#include "rope/io/csv_reader.h"

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace rope::io {

// One hour of derived driver features.
struct DriverRow {
    TimePoint tp;
    float f10, kp;
    float t1, t2, t3, t4;  // sin/cos harmonic features
    float doy;              // continuous = int_doy + hour/24
    int   hour_int;
};

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
    SpaceWeatherDB(std::vector<TimePoint> times,
                   std::vector<float>     f10,
                   std::vector<float>     kp,
                   std::vector<float>     doy,
                   std::vector<int>       hour);

    std::vector<TimePoint> times_;
    std::vector<float>     f10_, kp_, doy_;
    std::vector<int>       hour_;

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
