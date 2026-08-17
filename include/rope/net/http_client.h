#pragma once
#include <memory>
#include <string>

namespace rope::net {

// Minimal HTTPS GET boundary; virtual so callers (DriverCacheManager) can inject a fake in tests. No transitive TLS/HTTP dependency here — see http_client.cpp for the production implementation.
class IHttpClient {
public:
    virtual ~IHttpClient() = default;

    // Returns the response body; throws std::runtime_error on any failure (DNS, TLS, timeout, non-2xx status, empty body).
    virtual std::string get(const std::string& url) = 0;
};

// Production client (cpp-httplib + OpenSSL, bundled CA store). Only linked where actually called — see rope_net's CMake scoping.
std::unique_ptr<IHttpClient> make_http_client();

} // namespace rope::net
