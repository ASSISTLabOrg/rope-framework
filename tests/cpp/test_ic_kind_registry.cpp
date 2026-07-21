#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "../../src/forecast/backends/ic_source_factory.h"

TEST_CASE("known_ic_kinds() exactly matches stable entries in rope-registry/ic_kinds.json") {
    std::string ic_kinds_path = std::string(ROPE_REGISTRY_DIR) + "/ic_kinds.json";
    std::ifstream f(ic_kinds_path);
    REQUIRE(f.is_open());

    auto j = nlohmann::json::parse(f);

    std::vector<std::string> stable;
    for (const auto& entry : j) {
        if (entry.at("status").get<std::string>() == "stable")
            stable.push_back(entry.at("kind").get<std::string>());
    }
    std::sort(stable.begin(), stable.end());

    auto known = rope::forecast::known_ic_kinds();
    std::sort(known.begin(), known.end());

    CHECK(known == stable);
}
