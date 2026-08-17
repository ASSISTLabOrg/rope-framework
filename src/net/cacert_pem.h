#pragma once

namespace rope::net::detail {
// Bundled Mozilla CA root store (cmake/cacert.pem), embedded at configure time so TLS verification doesn't depend on the host's trust store.
extern const char* const cacert_pem;
} // namespace rope::net::detail
