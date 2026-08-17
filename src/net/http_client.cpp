#include "rope/net/http_client.h"
#include "cacert_pem.h"

#include <httplib.h>

#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
#error "rope_net requires cpp-httplib built with OpenSSL support (vcpkg cpp-httplib[openssl] feature)"
#endif

#include <cstring>
#include <stdexcept>
#include <string_view>

namespace rope::net {

namespace {

struct ParsedUrl {
    std::string scheme_host; // "https://host[:port]", passed to httplib::Client's constructor
    std::string path;        // everything from the first '/' after the host onward; "/" if none
};

// Splits scheme+host (for the Client constructor) from path+query (for Get()) — httplib::Client takes only the former.
ParsedUrl parse_url(const std::string& url) {
    std::string_view sv{url};
    std::string_view scheme;
    if (sv.starts_with("https://"))      scheme = "https://";
    else if (sv.starts_with("http://"))  scheme = "http://";
    else throw std::runtime_error("http_client: unsupported URL scheme in '" + url + "'");

    std::string_view rest = sv.substr(scheme.size());
    auto slash = rest.find('/');
    std::string_view host = (slash == std::string_view::npos) ? rest : rest.substr(0, slash);
    if (host.empty())
        throw std::runtime_error("http_client: empty host in URL '" + url + "'");

    std::string path = (slash == std::string_view::npos) ? "/" : std::string(rest.substr(slash));
    return ParsedUrl{std::string(scheme) + std::string(host), std::move(path)};
}

// cpp-httplib client backed by the bundled CA store — never touches the host's own trust store.
class CppHttplibClient final : public IHttpClient {
public:
    std::string get(const std::string& url) override {
        ParsedUrl u = parse_url(url);

        httplib::Client cli(u.scheme_host);
        cli.set_connection_timeout(10, 0);
        cli.set_read_timeout(30, 0);
        cli.set_follow_location(true);
        cli.set_default_headers({{"User-Agent", "rope-framework driver-cache"}});
        cli.enable_server_certificate_verification(true);
        cli.load_ca_cert_store(detail::cacert_pem, std::strlen(detail::cacert_pem));
        cli.enable_system_ca(false); // bundled store is authoritative regardless of host trust-store state

        auto res = cli.Get(u.path);
        if (!res)
            throw std::runtime_error(
                "http_client: GET " + url + " failed: " + httplib::to_string(res.error()));
        if (res->status < 200 || res->status >= 300)
            throw std::runtime_error(
                "http_client: GET " + url + " returned HTTP " + std::to_string(res->status));
        if (res->body.empty())
            throw std::runtime_error("http_client: GET " + url + " returned an empty body");

        return res->body;
    }
};

} // namespace

std::unique_ptr<IHttpClient> make_http_client() {
    return std::make_unique<CppHttplibClient>();
}

} // namespace rope::net
