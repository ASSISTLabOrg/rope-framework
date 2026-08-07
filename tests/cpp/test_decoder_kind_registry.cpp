#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "../../src/forecast/backends/decoder_factory.h"

TEST_CASE("known_decoder_kinds() exactly matches stable entries in rope-registry/decoder_kinds.json") {
    std::string decoder_kinds_path = std::string(ROPE_REGISTRY_DIR) + "/decoder_kinds.json";
    std::ifstream f(decoder_kinds_path);
    REQUIRE(f.is_open());

    auto j = nlohmann::json::parse(f);

    std::vector<std::string> stable;
    for (const auto& entry : j) {
        if (entry.at("status").get<std::string>() == "stable")
            stable.push_back(entry.at("kind").get<std::string>());
    }
    std::sort(stable.begin(), stable.end());

    auto known = rope::forecast::known_decoder_kinds();
    std::sort(known.begin(), known.end());

    CHECK(known == stable);
}
