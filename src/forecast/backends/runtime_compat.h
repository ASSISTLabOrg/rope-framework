#pragma once
#include "rope/io/model_manifest.h"

namespace rope::forecast {

void check_runtime_compat(const io::RuntimeRequirements& reqs);

} // namespace rope::forecast
