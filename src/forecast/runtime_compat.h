#pragma once
#include "rope/io/model_manifest.h"

namespace rope::forecast {

// Throws std::runtime_error if any backend listed in reqs has a major.minor
// version that doesn't match the currently linked runtime library.
// Empty version strings in reqs mean "not required"; those backends are skipped.
void check_runtime_compat(const io::RuntimeRequirements& reqs);

} // namespace rope::forecast
