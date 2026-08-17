#pragma once
#include <memory>
#include <string>

namespace rope::net {

// HTTPS GET boundary; virtual for test injection. Production implementation: http_client.cpp.
class IHttpClient {
public:
    virtual ~IHttpClient() = default;

    // Returns the response body; throws std::runtime_error on any failure (DNS, TLS, timeout, non-2xx status, empty body).
    virtual std::string get(const std::string& url) = 0;
};

// Production client: cpp-httplib + OpenSSL, bundled CA store.
std::unique_ptr<IHttpClient> make_http_client();

} // namespace rope::net
