#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "rope/io/ic_table.h"
#include "rope/io/ic_bin.h"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace Catch::Matchers;
namespace fs = std::filesystem;

// y1 = F10 + Kp, y2 = F10 - Kp — affine in both axes, so bilinear
// interpolation must reproduce them exactly at any point in the grid.
static fs::path make_ic_csv() {
    auto path = fs::temp_directory_path() / "rope_test_ic_table.csv";
    std::ofstream f(path);
    f << "F10,Kp,y1,y2\n"
         "100,0,100,100\n"
         "100,4,104,96\n"
         "200,0,200,200\n"
         "200,4,204,196\n";
    return path;
}

// Same shape, but axis names unrelated to F10/Kp — proves ICTable doesn't
// hardcode any particular axis identity.
static fs::path make_ic_csv_custom_axes() {
    auto path = fs::temp_directory_path() / "rope_test_ic_table_custom_axes.csv";
    std::ofstream f(path);
    f << "ap,dst,y1\n"
         "10,0,10\n"
         "10,4,14\n"
         "20,0,20\n"
         "20,4,24\n";
    return path;
}

TEST_CASE("ICTable: loads CSV and detects latent_dim from y1/y2 columns") {
    rope::io::ICTable table(make_ic_csv());
    CHECK(table.latent_dim() == 2);
}

TEST_CASE("ICTable: auto-detects axis names from non-'y<N>' CSV headers") {
    rope::io::ICTable table(make_ic_csv());
    auto axes = table.axis_names();
    REQUIRE(axes.size() == 2);
    CHECK(((axes[0] == "F10" && axes[1] == "Kp") || (axes[0] == "Kp" && axes[1] == "F10")));
}

TEST_CASE("ICTable: bilinear interpolation at an interior point matches the affine function") {
    rope::io::ICTable table(make_ic_csv());
    auto out = table.get_latent_coeffs(std::vector<float>{150.0f, 2.0f});
    REQUIRE(out.size() == 2);
    CHECK_THAT(out[0], WithinAbs(152.0f, 1e-3f));  // F10+Kp
    CHECK_THAT(out[1], WithinAbs(148.0f, 1e-3f));  // F10-Kp
}

TEST_CASE("ICTable: query at the grid's max corner returns the exact table value") {
    rope::io::ICTable table(make_ic_csv());
    auto out = table.get_latent_coeffs(std::vector<float>{200.0f, 4.0f});
    REQUIRE(out.size() == 2);
    CHECK_THAT(out[0], WithinAbs(204.0f, 1e-3f));
    CHECK_THAT(out[1], WithinAbs(196.0f, 1e-3f));
}

TEST_CASE("ICTable: query outside the convex hull falls back to nearest neighbour") {
    rope::io::ICTable table(make_ic_csv());
    // (50, -10) is closest to the (100, 0) corner.
    auto out = table.get_latent_coeffs(std::vector<float>{50.0f, -10.0f});
    REQUIRE(out.size() == 2);
    CHECK_THAT(out[0], WithinAbs(100.0f, 1e-3f));
    CHECK_THAT(out[1], WithinAbs(100.0f, 1e-3f));
}

TEST_CASE("ICTable: wrong number of axis values throws") {
    rope::io::ICTable table(make_ic_csv());
    REQUIRE_THROWS_AS(
        table.get_latent_coeffs(std::vector<float>{1.0f}),
        std::runtime_error
    );
}

TEST_CASE("ICTable: missing y1 column throws") {
    auto path = fs::temp_directory_path() / "rope_test_ic_table_no_y1.csv";
    std::ofstream f(path);
    f << "F10,Kp\n100,0\n";
    REQUIRE_THROWS_AS(rope::io::ICTable{path}, std::runtime_error);
}

TEST_CASE("ICTable: more than 2 non-y axis columns throws") {
    auto path = fs::temp_directory_path() / "rope_test_ic_table_3_axes.csv";
    std::ofstream f(path);
    f << "F10,Kp,ap,y1\n100,0,5,100\n";
    REQUIRE_THROWS_AS(rope::io::ICTable{path}, std::runtime_error);
}

TEST_CASE("ICTable: arbitrary (non-F10/Kp) axis names work end-to-end") {
    rope::io::ICTable table(make_ic_csv_custom_axes());
    REQUIRE(table.axis_names().size() == 2);
    auto out = table.get_latent_coeffs(std::vector<float>{15.0f, 2.0f});
    REQUIRE(out.size() == 1);
    CHECK_THAT(out[0], WithinAbs(17.0f, 1e-3f));
}

// ---------------------------------------------------------------------------
// IcBin (.icbin) — round trip + malformed file handling
// ---------------------------------------------------------------------------

TEST_CASE("IcBin: save/load round-trips through the binary format") {
    rope::io::ICTable table(make_ic_csv());
    auto bin = fs::temp_directory_path() / "rope_test_ic_table.icbin";
    rope::io::IcBin::save(table, bin);

    auto loaded = rope::io::IcBin::load(bin);
    CHECK(loaded.latent_dim() == table.latent_dim());
    CHECK(loaded.axis_names() == table.axis_names());
    auto out = loaded.get_latent_coeffs(std::vector<float>{200.0f, 4.0f});
    REQUIRE(out.size() == 2);
    CHECK_THAT(out[0], WithinAbs(204.0f, 1e-3f));
}

TEST_CASE("ICTable: from_file dispatches .icbin extension to the binary loader") {
    rope::io::ICTable table(make_ic_csv());
    auto bin = fs::temp_directory_path() / "rope_test_ic_table_dispatch.icbin";
    rope::io::IcBin::save(table, bin);

    auto loaded = rope::io::ICTable::from_file(bin);
    CHECK(loaded.latent_dim() == table.latent_dim());
}

static void write_name(std::ofstream& f, const std::string& name) {
    std::uint32_t len = static_cast<std::uint32_t>(name.size());
    f.write(reinterpret_cast<const char*>(&len), 4);
    f.write(name.data(), static_cast<std::streamsize>(len));
}

TEST_CASE("IcBin: bad magic throws") {
    auto path = fs::temp_directory_path() / "rope_test_ic_bad_magic.icbin";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    std::uint32_t bad_magic = 0xDEADBEEFu, version = 2u, nrows = 0u, latent_dim = 1u;
    f.write(reinterpret_cast<const char*>(&bad_magic),  4);
    f.write(reinterpret_cast<const char*>(&version),    4);
    f.write(reinterpret_cast<const char*>(&nrows),      4);
    f.write(reinterpret_cast<const char*>(&latent_dim), 4);
    f.close();

    REQUIRE_THROWS_AS(rope::io::IcBin::load(path), std::runtime_error);
}

TEST_CASE("IcBin: unsupported version throws") {
    auto path = fs::temp_directory_path() / "rope_test_ic_bad_version.icbin";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    std::uint32_t magic = 0x52504943u, bad_version = 99u, nrows = 0u, latent_dim = 1u;
    f.write(reinterpret_cast<const char*>(&magic),       4);
    f.write(reinterpret_cast<const char*>(&bad_version), 4);
    f.write(reinterpret_cast<const char*>(&nrows),       4);
    f.write(reinterpret_cast<const char*>(&latent_dim),  4);
    f.close();

    REQUIRE_THROWS_AS(rope::io::IcBin::load(path), std::runtime_error);
}

TEST_CASE("IcBin: latent_dim of zero throws") {
    auto path = fs::temp_directory_path() / "rope_test_ic_zero_latent_dim.icbin";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    std::uint32_t magic = 0x52504943u, version = 2u, nrows = 0u, latent_dim = 0u;
    f.write(reinterpret_cast<const char*>(&magic),      4);
    f.write(reinterpret_cast<const char*>(&version),    4);
    f.write(reinterpret_cast<const char*>(&nrows),      4);
    f.write(reinterpret_cast<const char*>(&latent_dim), 4);
    f.close();

    REQUIRE_THROWS_AS(rope::io::IcBin::load(path), std::runtime_error);
}

TEST_CASE("IcBin: truncated record data throws") {
    auto path = fs::temp_directory_path() / "rope_test_ic_truncated.icbin";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    std::uint32_t magic = 0x52504943u, version = 2u, nrows = 5u, latent_dim = 2u;
    f.write(reinterpret_cast<const char*>(&magic),      4);
    f.write(reinterpret_cast<const char*>(&version),    4);
    f.write(reinterpret_cast<const char*>(&nrows),      4);  // claims 5 rows
    f.write(reinterpret_cast<const char*>(&latent_dim), 4);
    write_name(f, "f10");
    write_name(f, "kp");
    // ... but no record data follows.
    f.close();

    REQUIRE_THROWS_AS(rope::io::IcBin::load(path), std::runtime_error);
}

TEST_CASE("IcBin: nonexistent file throws") {
    REQUIRE_THROWS_AS(
        rope::io::IcBin::load(fs::path("does_not_exist.icbin")),
        std::runtime_error
    );
}
