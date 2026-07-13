#include "rope/client/client.h"
#include "core/base64.h"
#include "core/wire.h"

#include "rope/core/datetime.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

namespace rope::client {

// ---------------------------------------------------------------------------
// Pimpl implementation
// ---------------------------------------------------------------------------

struct IpcClient::Impl {
    std::unique_ptr<platform::IpcSocket> sock;

    json call(const json& req) {
        wire::send_msg(*sock, req);
        json resp = wire::recv_msg(*sock);
        if (resp.value("status", "") != "ok") {
            std::string msg = resp.value("message", "(no message)");
            throw std::runtime_error("server error: " + msg);
        }
        return resp;
    }
};

// ---------------------------------------------------------------------------
// IpcClient public API
// ---------------------------------------------------------------------------

IpcClient::IpcClient(const std::filesystem::path& socket_path)
    : impl_(std::make_unique<Impl>())
{
    impl_->sock = platform::IpcSocket::connect(socket_path);
}

IpcClient::~IpcClient() = default;

ForecastResult IpcClient::forecast(const std::string& start_iso, int horizon) {
    json resp = impl_->call({
        {"type",    "forecast"},
        {"start",   start_iso},
        {"horizon", horizon}
    });
    return {resp.at("window_start"), resp.at("window_end")};
}

QueryResult IpcClient::get(const std::string& mode,
                            const std::string& time_iso,
                            double lst, double lat, double alt_km) {
    json resp = impl_->call({
        {"type", "get"},
        {"mode", mode},
        {"time", time_iso},
        {"lst",  lst},
        {"lat",  lat},
        {"alt",  alt_km}
    });
    return {resp.at("density"), resp.at("uncertainty")};
}

std::vector<QueryResult> IpcClient::batch_get(const std::string& mode,
                                               const std::vector<BatchPoint>& pts) {
    json points = json::array();
    for (const auto& p : pts) {
        points.push_back({
            {"time",   p.time_iso},
            {"lst",    p.lst},
            {"lat",    p.lat},
            {"alt",    p.alt_km}
        });
    }
    json resp = impl_->call({
        {"type",   "batch_get"},
        {"mode",   mode},
        {"points", std::move(points)}
    });

    std::vector<QueryResult> out;
    out.reserve(pts.size());
    for (const auto& r : resp.at("results"))
        out.push_back({r.at("density"), r.at("uncertainty")});
    return out;
}

ForecastGrid IpcClient::fetch_grid() {
    json resp = impl_->call({{"type", "fetch_grid"}});

    int H = resp.at("H").get<int>();

    // Decode base64 blobs into float32 arrays
    auto decode_field = [&](const std::string& key) -> std::vector<float> {
        auto bytes = rope::detail::base64_decode(resp.at(key).get<std::string>());
        std::size_t n_floats = bytes.size() / sizeof(float);
        std::vector<float> out(n_floats);
        std::memcpy(out.data(), bytes.data(), n_floats * sizeof(float));
        return out;
    };

    std::vector<float> density     = decode_field("density");
    std::vector<float> uncertainty = decode_field("uncertainty");

    // Parse datetimes to Unix timestamps
    std::vector<std::int64_t> times;
    times.reserve(static_cast<std::size_t>(H));
    for (const auto& s : resp.at("datetimes"))
        times.push_back(parse_datetime(s.get<std::string>()));

    ForecastGrid grid;
    grid.H           = H;
    grid.times       = std::move(times);
    grid.density     = std::move(density);
    grid.uncertainty = std::move(uncertainty);
    return grid;
}

void IpcClient::exit_server() {
    impl_->call({{"type", "exit"}});
}

} // namespace rope::client
