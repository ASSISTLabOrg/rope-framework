#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "../../src/forecast/pipeline_registry.h"

TEST_CASE("known_kinds() exactly matches stable entries in rope-registry/kinds.json") {
    std::string kinds_path = std::string(ROPE_REGISTRY_DIR) + "/kinds.json";
    std::ifstream f(kinds_path);
    REQUIRE(f.is_open());

    auto j = nlohmann::json::parse(f);

    std::vector<std::string> stable;
    for (const auto& entry : j) {
        if (entry.at("status").get<std::string>() == "stable")
            stable.push_back(entry.at("kind").get<std::string>());
    }
    std::sort(stable.begin(), stable.end());

    auto known = rope::forecast::known_kinds();
    std::sort(known.begin(), known.end());

    CHECK(known == stable);
}
