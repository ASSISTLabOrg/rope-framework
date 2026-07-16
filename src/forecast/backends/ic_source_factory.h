#pragma once
// Constructs an IICSource from manifest.ic.kind.

#include "ic_source.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace rope::forecast {

// Throws std::runtime_error for an unrecognized ic_kind.
std::unique_ptr<IICSource> make_ic_source(
    const std::filesystem::path& dir,
    const std::string&           ic_kind);

inline std::vector<std::string> known_ic_kinds() {
    return {"ic_lookup_table"};
}

} // namespace rope::forecast
