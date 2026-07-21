// Drift test: rope-registry fixtures vs ModelManifest::load().

#include <catch2/catch_test_macros.hpp>
#include "rope/io/model_manifest.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

using rope::io::ModelManifest;
namespace fs = std::filesystem;

static fs::path copy_fixture_to_tempdir(const char* fixture_name) {
    fs::path src = fs::path(ROPE_REGISTRY_DIR) / "tests" / "fixtures" / fixture_name;
    std::ifstream in(src, std::ios::binary);
    if (!in)
        throw std::runtime_error("copy_fixture_to_tempdir: cannot open " + src.string());
    std::ostringstream ss;
    ss << in.rdbuf();

    fs::path dir = fs::temp_directory_path() / (std::string("rope_schema_fixture_") + fixture_name);
    fs::create_directories(dir);
    std::ofstream out(dir / "model_manifest.json", std::ios::binary);
    out << ss.str();
    return dir;
}

TEST_CASE("ModelManifest::load agrees with rope-registry: valid_manifest.json") {
    CHECK_NOTHROW(ModelManifest::load(copy_fixture_to_tempdir("valid_manifest.json")));
}

TEST_CASE("ModelManifest::load agrees with rope-registry: valid_manifest_with_validation.json") {
    CHECK_NOTHROW(ModelManifest::load(copy_fixture_to_tempdir("valid_manifest_with_validation.json")));
}

TEST_CASE("ModelManifest::load agrees with rope-registry: invalid_ic_missing_grid_axes.json throws") {
    CHECK_THROWS_AS(
        ModelManifest::load(copy_fixture_to_tempdir("invalid_ic_missing_grid_axes.json")),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest::load agrees with rope-registry: invalid_envelope_missing_ic.json throws") {
    CHECK_THROWS_AS(
        ModelManifest::load(copy_fixture_to_tempdir("invalid_envelope_missing_ic.json")),
        std::runtime_error
    );
}

TEST_CASE("ModelManifest::load intentionally does not validate ic.kind membership (deferred to make_ic_source)") {
    CHECK_NOTHROW(ModelManifest::load(copy_fixture_to_tempdir("invalid_ic_bad_kind.json")));
}
