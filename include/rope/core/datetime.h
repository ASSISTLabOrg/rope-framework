#pragma once
// UTC datetime utilities built on C++20 <chrono>; no platform-specific headers.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

namespace rope {

using TimePoint = std::int64_t;   // UTC seconds since Unix epoch

namespace detail {

inline TimePoint ymd_hms_to_unix(int y, int mo, int d, int h, int mi, int s) {
    using namespace std::chrono;
    auto ymd = year{y} / month{static_cast<unsigned>(mo)} / day{static_cast<unsigned>(d)};
    auto tp   = sys_days{ymd} + hours{h} + minutes{mi} + seconds{s};
    return static_cast<TimePoint>(tp.time_since_epoch().count());
}

} // namespace detail

// Parse "YYYY-MM-DDTHH:MM:SS" or "YYYY-MM-DD HH:MM:SS" (UTC).
inline TimePoint parse_datetime(std::string_view sv) {
    int y, mo, d, h, mi, s;
    std::string str{sv};
    for (char& c : str) if (c == 'T') c = ' ';   // normalise ISO 8601 separator
    if (std::sscanf(str.c_str(), "%d-%d-%d %d:%d:%d", &y, &mo, &d, &h, &mi, &s) != 6)
        throw std::runtime_error("datetime: bad format: " + std::string(sv));
    return detail::ymd_hms_to_unix(y, mo, d, h, mi, s);
}

// Format TimePoint as "YYYY-MM-DDTHH:MM:SS" (UTC, ISO 8601).
inline std::string format_iso(TimePoint tp) {
    using namespace std::chrono;
    auto sys_tp = sys_seconds{seconds{tp}};
    auto dp     = floor<days>(sys_tp);
    year_month_day ymd{dp};
    hh_mm_ss<seconds> hms{sys_tp - dp};
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d",
        static_cast<int>(ymd.year()),
        static_cast<unsigned>(ymd.month()),
        static_cast<unsigned>(ymd.day()),
        static_cast<int>(hms.hours().count()),
        static_cast<int>(hms.minutes().count()),
        static_cast<int>(hms.seconds().count()));
    return buf;
}

// Floor to the nearest whole hour.
inline TimePoint floor_hour(TimePoint tp) noexcept {
    return (tp / 3600) * 3600;
}

// Extract UTC hour [0,23], day-of-year [1,366], and year from a TimePoint.
inline void unpack(TimePoint tp, int& hour, int& doy, int& year) {
    using namespace std::chrono;
    auto sys_tp = sys_seconds{seconds{tp}};
    auto dp     = floor<days>(sys_tp);
    year_month_day ymd{dp};
    hh_mm_ss<seconds> hms{sys_tp - dp};
    year = static_cast<int>(ymd.year());
    hour = static_cast<int>(hms.hours().count());
    auto jan1 = sys_days{ymd.year() / January / 1};
    doy = static_cast<int>((dp - jan1).count()) + 1;
}

// Compute the four harmonic driver features and continuous day-of-year.
//   t1 = sin(2π·hour/24),    t2 = cos(2π·hour/24)
//   t3 = sin(2π·doy/365.25), t4 = cos(2π·doy/365.25)
inline void harmonics(TimePoint tp,
                      float& t1, float& t2, float& t3, float& t4,
                      float& cont_doy) {
    int h, doy, yr;
    unpack(tp, h, doy, yr);
    constexpr double TWO_PI = 2.0 * 3.14159265358979323846;
    t1 = static_cast<float>(std::sin(TWO_PI * h / 24.0));
    t2 = static_cast<float>(std::cos(TWO_PI * h / 24.0));
    t3 = static_cast<float>(std::sin(TWO_PI * doy / 365.25));
    t4 = static_cast<float>(std::cos(TWO_PI * doy / 365.25));
    cont_doy = static_cast<float>(doy + h / 24.0);
}

} // namespace rope
