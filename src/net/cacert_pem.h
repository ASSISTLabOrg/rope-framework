#pragma once

namespace rope::net::detail {
// Bundled Mozilla CA root store (cmake/cacert.pem), embedded at configure time.
extern const char* const cacert_pem;
} // namespace rope::net::detail
