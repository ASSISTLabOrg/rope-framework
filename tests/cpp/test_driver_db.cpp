#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "rope/io/driver_db.h"
#include "rope/io/driver_bin.h"
#include "rope/core/datetime.h"
#include <cstdint>
#include <filesystem>
#include <fstream>

using namespace Catch::Matchers;
namespace fs = std::filesystem;

// Six consecutive hourly rows, 2024-01-01T00:00:00 .. 05:00:00.
static fs::path make_driver_csv() {
    auto path = fs::temp_directory_path() / "rope_test_driver_db.csv";
    std::ofstream f(path);
    f << "datetime,f10,kp\n"
         "2024-01-01T00:00:00,120.0,2.0\n"
         "2024-01-01T01:00:00,121.0,2.1\n"
         "2024-01-01T02:00:00,122.0,2.2\n"
         "2024-01-01T03:00:00,123.0,2.3\n"
         "2024-01-01T04:00:00,124.0,2.4\n"
         "2024-01-01T05:00:00,125.0,2.5\n";
    return path;
}

TEST_CASE("SpaceWeatherDB: loads CSV and reports size and time range") {
    auto db = rope::io::SpaceWeatherDB::from_file(make_driver_csv());
    CHECK(db.size() == 6);
    CHECK(db.time_min() == rope::parse_datetime("2024-01-01T00:00:00"));
    CHECK(db.time_max() == rope::parse_datetime("2024-01-01T05:00:00"));
}

TEST_CASE("SpaceWeatherDB: lookup returns the matching row") {
    auto db = rope::io::SpaceWeatherDB::from_file(make_driver_csv());
    auto row = db.lookup(rope::parse_datetime("2024-01-01T02:00:00"));
    CHECK_THAT(row.get("f10"), WithinAbs(122.0f, 1e-4f));
    CHECK_THAT(row.get("kp"),  WithinAbs(2.2f,   1e-4f));
    CHECK(row.get("hour_int") == 2.0f);
}

TEST_CASE("SpaceWeatherDB: loads an arbitrary extra raw column by name") {
    auto path = fs::temp_directory_path() / "rope_test_driver_db_extra.csv";
    std::ofstream f(path);
    f << "datetime,kp,ap\n"
         "2024-01-01T00:00:00,2.0,7.5\n"
         "2024-01-01T01:00:00,2.1,8.0\n";
    f.close();

    auto db  = rope::io::SpaceWeatherDB::from_file(path);
    auto row = db.lookup(rope::parse_datetime("2024-01-01T01:00:00"));
    CHECK_THAT(row.get("kp"), WithinAbs(2.1f, 1e-4f));
    CHECK_THAT(row.get("ap"), WithinAbs(8.0f, 1e-4f));
    REQUIRE_THROWS_AS(row.get("f10"), std::runtime_error);
}

TEST_CASE("SpaceWeatherDB: CSV header order does not matter") {
    auto path = fs::temp_directory_path() / "rope_test_driver_db_shuffled.csv";
    std::ofstream f(path);
    f << "kp,datetime,f10\n"
         "2.2,2024-01-01T02:00:00,122.0\n";
    f.close();

    auto db  = rope::io::SpaceWeatherDB::from_file(path);
    auto row = db.lookup(rope::parse_datetime("2024-01-01T02:00:00"));
    CHECK_THAT(row.get("f10"), WithinAbs(122.0f, 1e-4f));
    CHECK_THAT(row.get("kp"),  WithinAbs(2.2f,   1e-4f));
}

TEST_CASE("SpaceWeatherDB: a raw CSV column literally named t1-t4 throws") {
    auto path = fs::temp_directory_path() / "rope_test_driver_db_reserved.csv";
    std::ofstream f(path);
    f << "datetime,f10,t1\n2024-01-01T00:00:00,120.0,0.5\n";
    f.close();

    REQUIRE_THROWS_AS(rope::io::SpaceWeatherDB::from_file(path), std::runtime_error);
}

TEST_CASE("SpaceWeatherDB: f10_41day_avg averages the trailing window (fewer rows near coverage start)") {
    // 10 hourly rows, f10 = 100, 101, ..., 109. Window is 984h, so every row here
    // is a "partial window" case -- average of all rows from the start through idx.
    auto path = fs::temp_directory_path() / "rope_test_driver_db_f10avg.csv";
    std::ofstream f(path);
    f << "datetime,f10\n";
    for (int i = 0; i < 10; ++i)
        f << "2024-01-01T" << (i < 10 ? "0" : "") << i << ":00:00," << (100.0 + i) << "\n";
    f.close();

    auto db = rope::io::SpaceWeatherDB::from_file(path);

    // idx 0: avg of just [100] = 100.
    auto row0 = db.lookup(rope::parse_datetime("2024-01-01T00:00:00"));
    CHECK_THAT(row0.get("f10_41day_avg"), WithinAbs(100.0f, 1e-4f));

    // idx 9 (last row): avg of [100..109] = 104.5.
    auto row9 = db.lookup(rope::parse_datetime("2024-01-01T09:00:00"));
    CHECK_THAT(row9.get("f10_41day_avg"), WithinAbs(104.5f, 1e-4f));
}

TEST_CASE("SpaceWeatherDB: f10_41day_avg window caps at 984 hours once enough history exists") {
    // 985 hourly rows, f10 = 0, 1, 2, ..., 984. At the last row (idx 984), the
    // trailing 984-hour window is rows [1..984], excluding row 0 -- average
    // should be (1+984)/2 = 492.5, not (0+984)/2 = 492.
    auto path = fs::temp_directory_path() / "rope_test_driver_db_f10avg_cap.csv";
    std::ofstream f(path);
    f << "datetime,f10\n";
    auto t0 = rope::parse_datetime("2024-01-01T00:00:00");
    for (int i = 0; i <= 984; ++i)
        f << rope::format_iso(t0 + static_cast<rope::TimePoint>(i) * 3600) << "," << i << "\n";
    f.close();

    auto db  = rope::io::SpaceWeatherDB::from_file(path);
    auto row = db.lookup(t0 + static_cast<rope::TimePoint>(984) * 3600);
    CHECK_THAT(row.get("f10_41day_avg"), WithinAbs(492.5f, 1e-3f));
}

TEST_CASE("SpaceWeatherDB: f10_41day_avg throws if requested but no raw 'f10' column exists") {
    auto path = fs::temp_directory_path() / "rope_test_driver_db_f10avg_missing.csv";
    std::ofstream f(path);
    f << "datetime,kp\n2024-01-01T00:00:00,2.0\n";
    f.close();

    auto db  = rope::io::SpaceWeatherDB::from_file(path);
    auto row = db.lookup(rope::parse_datetime("2024-01-01T00:00:00"));
    REQUIRE_THROWS_AS(row.get("f10_41day_avg"), std::runtime_error);
}

TEST_CASE("SpaceWeatherDB: a raw CSV column literally named f10_41day_avg is used directly, not recomputed") {
    auto path = fs::temp_directory_path() / "rope_test_driver_db_f10avg_direct.csv";
    std::ofstream f(path);
    // 115.0 deliberately differs from what compute_f10_41day_avg would give for this single row (120.0).
    f << "datetime,f10,f10_41day_avg\n2024-01-01T00:00:00,120.0,115.0\n";
    f.close();

    auto db  = rope::io::SpaceWeatherDB::from_file(path);
    auto row = db.lookup(rope::parse_datetime("2024-01-01T00:00:00"));
    CHECK_THAT(row.get("f10_41day_avg"), WithinAbs(115.0f, 1e-4f));
}

TEST_CASE("SpaceWeatherDB: f10_41day_avg can be supplied directly without any raw 'f10' column") {
    auto path = fs::temp_directory_path() / "rope_test_driver_db_f10avg_standalone.csv";
    std::ofstream f(path);
    f << "datetime,kp,f10_41day_avg\n2024-01-01T00:00:00,2.0,108.3\n";
    f.close();

    auto db  = rope::io::SpaceWeatherDB::from_file(path);
    auto row = db.lookup(rope::parse_datetime("2024-01-01T00:00:00"));
    CHECK_THAT(row.get("f10_41day_avg"), WithinAbs(108.3f, 1e-4f));
}

TEST_CASE("SpaceWeatherBin: f10_41day_avg round-trips as a raw column, used directly not recomputed") {
    auto path = fs::temp_directory_path() / "rope_test_driver_db_f10avg_swbin_src.csv";
    std::ofstream f(path);
    f << "datetime,f10,f10_41day_avg\n2024-01-01T00:00:00,120.0,115.0\n";
    f.close();

    auto db  = rope::io::SpaceWeatherDB::from_file(path);
    auto bin = fs::temp_directory_path() / "rope_test_driver_db_f10avg.swbin";
    rope::io::SpaceWeatherBin::save(db, bin);

    auto loaded = rope::io::SpaceWeatherBin::load(bin);
    auto row    = loaded.lookup(rope::parse_datetime("2024-01-01T00:00:00"));
    CHECK_THAT(row.get("f10_41day_avg"), WithinAbs(115.0f, 1e-4f));
}

TEST_CASE("SpaceWeatherDB: lookup at a missing timestamp throws") {
    auto db = rope::io::SpaceWeatherDB::from_file(make_driver_csv());
    REQUIRE_THROWS_AS(
        db.lookup(rope::parse_datetime("2024-01-01T00:30:00")),
        std::runtime_error
    );
}

TEST_CASE("DriverWindowBuilder: builds (seq_len-1+horizon) rows in chronological order") {
    auto db = rope::io::SpaceWeatherDB::from_file(make_driver_csv());
    auto rows = rope::io::DriverWindowBuilder::build(
        db, "2024-01-01T03:00:00", /*horizon=*/2, /*seq_len=*/3);

    REQUIRE(rows.size() == 4);  // (3-1) + 2
    CHECK(rows.front().tp == rope::parse_datetime("2024-01-01T01:00:00"));
    for (std::size_t i = 1; i < rows.size(); ++i)
        CHECK(rows[i].tp == rows[i - 1].tp + 3600);
}

TEST_CASE("DriverWindowBuilder: missing hourly slot throws") {
    auto db = rope::io::SpaceWeatherDB::from_file(make_driver_csv());
    // Window would need 2024-01-01T06:00:00, one hour past the last row.
    REQUIRE_THROWS_AS(
        rope::io::DriverWindowBuilder::build(
            db, "2024-01-01T05:00:00", /*horizon=*/2, /*seq_len=*/3),
        std::runtime_error
    );
}

// ---------------------------------------------------------------------------
// SpaceWeatherBin (.swbin) — round trip + malformed file handling
// ---------------------------------------------------------------------------

TEST_CASE("SpaceWeatherBin: save/load round-trips through the binary format") {
    auto db  = rope::io::SpaceWeatherDB::from_file(make_driver_csv());
    auto bin = fs::temp_directory_path() / "rope_test_driver_db.swbin";
    rope::io::SpaceWeatherBin::save(db, bin);

    auto loaded = rope::io::SpaceWeatherBin::load(bin);
    CHECK(loaded.size() == db.size());
    auto row = loaded.lookup(rope::parse_datetime("2024-01-01T02:00:00"));
    CHECK_THAT(row.get("f10"), WithinAbs(122.0f, 1e-4f));
}

TEST_CASE("SpaceWeatherDB: from_file dispatches .swbin extension to the binary loader") {
    auto db  = rope::io::SpaceWeatherDB::from_file(make_driver_csv());
    auto bin = fs::temp_directory_path() / "rope_test_driver_db_dispatch.swbin";
    rope::io::SpaceWeatherBin::save(db, bin);

    auto loaded = rope::io::SpaceWeatherDB::from_file(bin);
    CHECK(loaded.size() == db.size());
}

TEST_CASE("SpaceWeatherBin: bad magic throws") {
    auto path = fs::temp_directory_path() / "rope_test_sw_bad_magic.swbin";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    std::uint32_t bad_magic = 0xDEADBEEFu, version = 2u, nrows = 0u, ncols = 0u;
    f.write(reinterpret_cast<const char*>(&bad_magic), 4);
    f.write(reinterpret_cast<const char*>(&version),   4);
    f.write(reinterpret_cast<const char*>(&nrows),     4);
    f.write(reinterpret_cast<const char*>(&ncols),     4);
    f.close();

    REQUIRE_THROWS_AS(rope::io::SpaceWeatherBin::load(path), std::runtime_error);
}

TEST_CASE("SpaceWeatherBin: unsupported version throws") {
    auto path = fs::temp_directory_path() / "rope_test_sw_bad_version.swbin";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    std::uint32_t magic = 0x52505357u, bad_version = 99u, nrows = 0u, ncols = 0u;
    f.write(reinterpret_cast<const char*>(&magic),       4);
    f.write(reinterpret_cast<const char*>(&bad_version), 4);
    f.write(reinterpret_cast<const char*>(&nrows),       4);
    f.write(reinterpret_cast<const char*>(&ncols),       4);
    f.close();

    REQUIRE_THROWS_AS(rope::io::SpaceWeatherBin::load(path), std::runtime_error);
}

TEST_CASE("SpaceWeatherBin: truncated record data throws") {
    auto path = fs::temp_directory_path() / "rope_test_sw_truncated.swbin";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    std::uint32_t magic = 0x52505357u, version = 2u, nrows = 5u, ncols = 1u;
    f.write(reinterpret_cast<const char*>(&magic),    4);
    f.write(reinterpret_cast<const char*>(&version),  4);
    f.write(reinterpret_cast<const char*>(&nrows),    4);  // claims 5 rows
    f.write(reinterpret_cast<const char*>(&ncols),    4);
    {
        std::uint32_t name_len = 3;
        f.write(reinterpret_cast<const char*>(&name_len), 4);
        f.write("f10", 3);
    }
    // ... but no record data follows.
    f.close();

    REQUIRE_THROWS_AS(rope::io::SpaceWeatherBin::load(path), std::runtime_error);
}

TEST_CASE("SpaceWeatherBin: nonexistent file throws") {
    REQUIRE_THROWS_AS(
        rope::io::SpaceWeatherBin::load(fs::path("does_not_exist.swbin")),
        std::runtime_error
    );
}
