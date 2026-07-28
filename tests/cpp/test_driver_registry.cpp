#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "rope/io/driver_db.h"

TEST_CASE("known_derived_driver_names() exactly matches 'derived' entries in rope-registry/driver_registry.json") {
    std::string registry_path = std::string(ROPE_REGISTRY_DIR) + "/driver_registry.json";
    std::ifstream f(registry_path);
    REQUIRE(f.is_open());

    auto j = nlohmann::json::parse(f);

    std::vector<std::string> derived;
    for (const auto& entry : j) {
        if (entry.at("kind").get<std::string>() == "derived")
            derived.push_back(entry.at("name").get<std::string>());
    }
    std::sort(derived.begin(), derived.end());

    auto known = rope::io::known_derived_driver_names();
    std::sort(known.begin(), known.end());

    CHECK(known == derived);
}
