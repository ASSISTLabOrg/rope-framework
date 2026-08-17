#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "rope/io/driver_cache.h"
#include "rope/io/driver_db.h"
#include "rope/net/http_client.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

namespace fs = std::filesystem;
using Catch::Matchers::WithinAbs;

namespace {

// Test double for rope::net::IHttpClient — either returns a canned body or always throws,
// so refresh()'s success and failure paths are both exercisable offline.
class FakeHttpClient final : public rope::net::IHttpClient {
public:
    struct AlwaysFails {};

    explicit FakeHttpClient(std::string body) : body_(std::move(body)) {}
    FakeHttpClient(AlwaysFails, std::string message)
        : fails_(true), message_(std::move(message)) {}

    std::string get(const std::string& /*url*/) override {
        if (fails_) throw std::runtime_error(message_);
        return body_;
    }

private:
    bool fails_ = false;
    std::string body_;
    std::string message_;
};

std::unique_ptr<rope::net::IHttpClient> fake_ok(std::string body) {
    return std::make_unique<FakeHttpClient>(std::move(body));
}

std::unique_ptr<rope::net::IHttpClient> fake_failing(std::string message = "simulated network failure") {
    return std::make_unique<FakeHttpClient>(FakeHttpClient::AlwaysFails{}, std::move(message));
}

// Minimal 3-day CelesTrak-shaped CSV: only the columns convert_celestrak_csv_to_swbin() actually reads.
// KP values are raw tenths (CelesTrak convention); AP values are already linear (no rescaling); F10.7 is SFU.
const std::string kValidCelestrakCsv =
    "DATE,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,F10.7_OBS\n"
    "2024-01-01,10,13,17,20,23,27,30,33,4,5,6,7,9,12,15,18,150.0\n"
    "2024-01-02,13,17,20,23,27,30,33,37,5,6,7,9,12,15,18,22,152.5\n"
    "2024-01-03,17,20,23,27,30,33,37,40,6,7,9,12,15,18,22,27,155.0\n";

} // namespace

static fs::path make_cache_dir(const std::string& name) {
    auto dir = fs::temp_directory_path() / name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return dir;
}

static void write_dummy_file(const fs::path& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << "dummy";
}

TEST_CASE("known_sources: registry contains the documented CelesTrak sources") {
    const auto& sources = rope::io::known_sources();
    CHECK(sources.count("celestrak_sw")     == 1);
    CHECK(sources.count("celestrak_sw_all") == 1);
}

TEST_CASE("DriverCacheManager: unknown source throws") {
    auto dir = make_cache_dir("rope_test_cache_unknown");
    rope::io::DriverCacheManager mgr(dir, /*max_age_hours=*/24, fake_failing());
    REQUIRE_THROWS_AS(mgr.get_path("not_a_real_source"), std::runtime_error);
}

TEST_CASE("DriverCacheManager: no cached file and a failed refresh throws") {
    auto dir = make_cache_dir("rope_test_cache_missing");
    rope::io::DriverCacheManager mgr(dir, /*max_age_hours=*/24, fake_failing());
    // Nothing cached, and the injected client always fails — nothing to fall back
    // to, so this must fail loudly rather than return a bad path.
    REQUIRE_THROWS_AS(mgr.get_path("celestrak_sw"), std::runtime_error);
}

TEST_CASE("DriverCacheManager: stale cache falls back to the stale file when refresh fails") {
    auto dir  = make_cache_dir("rope_test_cache_stale");
    auto dest = dir / "celestrak_sw.swbin";
    write_dummy_file(dest);

    // Backdate the file well past max_age_hours so a refresh is attempted.
    fs::last_write_time(dest,
        fs::file_time_type::clock::now() - std::chrono::hours(48));

    rope::io::DriverCacheManager mgr(dir, /*max_age_hours=*/1, fake_failing());
    auto path = mgr.get_path("celestrak_sw");
    CHECK(path == dest);
    CHECK(fs::exists(path));
}

TEST_CASE("DriverCacheManager: fresh cache file is returned without refreshing") {
    auto dir  = make_cache_dir("rope_test_cache_fresh");
    auto dest = dir / "celestrak_sw.swbin";
    write_dummy_file(dest);  // mtime defaults to "now" — within max_age_hours

    // Fresh-file path never calls the client at all — a failing fake proves that.
    rope::io::DriverCacheManager mgr(dir, /*max_age_hours=*/24, fake_failing());
    auto path = mgr.get_path("celestrak_sw");
    CHECK(path == dest);
}

TEST_CASE("DriverCacheManager: successful refresh downloads, converts, and writes a usable .swbin") {
    auto dir = make_cache_dir("rope_test_cache_success");

    rope::io::DriverCacheManager mgr(dir, /*max_age_hours=*/24, fake_ok(kValidCelestrakCsv));
    auto path = mgr.get_path("celestrak_sw");
    REQUIRE(fs::exists(path));
    CHECK(path == dir / "celestrak_sw.swbin");

    auto db = rope::io::SpaceWeatherDB::from_file(path);

    // Knot points: PCHIP interpolation reproduces the source value exactly at each knot.
    auto row0 = db.lookup(rope::parse_datetime("2024-01-01 00:00:00"));
    CHECK_THAT(row0.get("f10"), WithinAbs(150.0, 1e-3));
    CHECK_THAT(row0.get("kp"),  WithinAbs(1.0, 1e-3));   // KP1 = 10 (tenths) -> 1.0
    CHECK_THAT(row0.get("ap"),  WithinAbs(4.0, 1e-3));   // AP1 = 4, no rescaling

    auto row1 = db.lookup(rope::parse_datetime("2024-01-02 00:00:00"));
    CHECK_THAT(row1.get("f10"), WithinAbs(152.5, 1e-3));
    CHECK_THAT(row1.get("kp"),  WithinAbs(1.3, 1e-3));   // KP1 = 13 (tenths) -> 1.3
    CHECK_THAT(row1.get("ap"),  WithinAbs(5.0, 1e-3));   // AP1 = 5, no rescaling

    // Re-fetching immediately must not re-download (file is now fresh) — same failing-fake trick as above.
    rope::io::DriverCacheManager mgr2(dir, /*max_age_hours=*/24, fake_failing());
    auto path2 = mgr2.get_path("celestrak_sw");
    CHECK(path2 == path);
}

TEST_CASE("convert_celestrak_csv_to_swbin: converts a local raw CelesTrak CSV with no network involved") {
    // Exercises the exact seam `rope convert-sw` uses for a raw (unconverted) CelesTrak file —
    // no DriverCacheManager, no IHttpClient, just the free converter on bytes already in hand.
    auto dir  = make_cache_dir("rope_test_celestrak_convert_direct");
    auto dest = dir / "local.swbin";

    rope::io::convert_celestrak_csv_to_swbin(kValidCelestrakCsv, dest);
    REQUIRE(fs::exists(dest));

    auto db = rope::io::SpaceWeatherDB::from_file(dest);
    // Day 3 (the last row) never gets an hourly entry — build_hourly_daily() needs a *following*
    // row to supply the 24h continuity endpoint, so only day 1 and day 2 are ever emitted.
    auto row = db.lookup(rope::parse_datetime("2024-01-02 00:00:00"));
    CHECK_THAT(row.get("f10"), WithinAbs(152.5, 1e-3));
    CHECK_THAT(row.get("kp"),  WithinAbs(1.3, 1e-3));   // KP1 = 13 (tenths) -> 1.3
    CHECK_THAT(row.get("ap"),  WithinAbs(5.0, 1e-3));   // AP1 = 5, no rescaling
}

TEST_CASE("DriverCacheManager: non-2xx / malformed response fails loudly and reaches get_path()") {
    auto dir = make_cache_dir("rope_test_cache_malformed");

    // Simulates the client surfacing an HTTP-level failure (e.g. non-2xx) as a thrown error —
    // exercised through get_path() so the get_path -> refresh -> download chain is proven, not
    // just convert_celestrak_csv_to_swbin() in isolation.
    rope::io::DriverCacheManager http_fail(dir, /*max_age_hours=*/24,
        fake_failing("http_client: GET ... returned HTTP 503"));
    REQUIRE_THROWS_AS(http_fail.get_path("celestrak_sw"), std::runtime_error);

    // Simulates a client that "succeeds" but returns a body convert_celestrak_csv_to_swbin() can't parse.
    rope::io::DriverCacheManager bad_body(dir, /*max_age_hours=*/24, fake_ok("not,a,celestrak,csv\n"));
    REQUIRE_THROWS_AS(bad_body.get_path("celestrak_sw"), std::runtime_error);
}

TEST_CASE("DriverCacheManager: CSV missing AP columns throws (both Kp and Ap are required raw columns)") {
    auto dir = make_cache_dir("rope_test_cache_missing_ap");
    const std::string kp_only_csv =
        "DATE,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,F10.7_OBS\n"
        "2024-01-01,10,13,17,20,23,27,30,33,150.0\n"
        "2024-01-02,13,17,20,23,27,30,33,37,152.5\n";

    rope::io::DriverCacheManager mgr(dir, /*max_age_hours=*/24, fake_ok(kp_only_csv));
    REQUIRE_THROWS_AS(mgr.get_path("celestrak_sw"), std::runtime_error);
}
