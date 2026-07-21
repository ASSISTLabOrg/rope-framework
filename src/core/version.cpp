// version.cpp — exposes the build-time version string via a stable C symbol.
// The version constants themselves live in the generated rope/core/version.h.

#include "rope/core/version.h"
#include "rope/core/export.h"

extern "C" ROPE_API const char* rope_version_string() {
    return rope::version::string();
}