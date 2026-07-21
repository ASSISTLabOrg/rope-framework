#include <catch2/catch_test_macros.hpp>
#include "rope/forecast/pipeline.h"
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

TEST_CASE("forecast::load: missing model_manifest.json throws") {
    rope::forecast::Config cfg;
    cfg.exported_dir = fs::path(ROPE_CPP_FIXTURE_DIR) / "does_not_exist";
    CHECK_THROWS_AS(rope::forecast::load(cfg), std::runtime_error);
}

TEST_CASE("forecast::load: mismatched onnxruntime version throws before loading models") {
    rope::forecast::Config cfg;
    cfg.exported_dir = fs::path(ROPE_CPP_FIXTURE_DIR) / "bad_ort";
    CHECK_THROWS_AS(rope::forecast::load(cfg), std::runtime_error);
}

TEST_CASE("forecast::load: unrecognized ic.kind throws") {
    rope::forecast::Config cfg;
    cfg.exported_dir = fs::path(ROPE_CPP_FIXTURE_DIR) / "bad_ic_kind";
    CHECK_THROWS_AS(rope::forecast::load(cfg), std::runtime_error);
}
