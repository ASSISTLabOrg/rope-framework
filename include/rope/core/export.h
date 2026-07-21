#pragma once
// Visibility macro for shared-library symbols. Define ROPE_BUILDING_SHARED when building the shared lib, ROPE_SHARED when linking against it; neither needed for static/header-only use.

#if defined(_WIN32)
#  if defined(ROPE_BUILDING_SHARED)
#    define ROPE_API __declspec(dllexport)
#  elif defined(ROPE_SHARED)
#    define ROPE_API __declspec(dllimport)
#  else
#    define ROPE_API
#  endif
#else
#  if defined(ROPE_BUILDING_SHARED)
#    define ROPE_API __attribute__((visibility("default")))
#  else
#    define ROPE_API
#  endif
#endif
