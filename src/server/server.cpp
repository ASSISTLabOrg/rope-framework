// Server implementation.
//
// Lifecycle:
//   cli spawns: rope --serve --socket-path <p> --config-path <p>
//   server::run() binds the socket, loads config, and enters the accept loop.
//   One connection is served at a time (synchronous request/response).
//   Shutdown on: "exit" request, SIGINT, or SIGTERM.

#include "rope/server/server.h"

#include "rope/core/datetime.h"
#include "rope/core/platform.h"
#include "rope/core/types.h"
#include "rope/interpolate/grid_interpolator.h"
#include "rope/io/config_reader.h"
#include "core/base64.h"
#include "core/wire.h"

#ifdef ROPE_HAS_FORECAST
#  include "rope/forecast/pipeline.h"
#endif

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

using json = nlohmann::json;

namespace rope::server {

// ---------------------------------------------------------------------------
// Signal handling — sets g_running to false on SIGINT / SIGTERM.
// std::atomic store is async-signal-safe.
// ---------------------------------------------------------------------------
static std::atomic<bool> g_running{false};

static void on_signal(int) noexcept { g_running = false; }

static json error_resp(const std::string& code, const std::string& msg) {
    return {{"status", "error"}, {"code", code}, {"message", msg}};
}

// ---------------------------------------------------------------------------
// Grid cache — one ForecastGrid + its interpolator.
// Replaced on each successful forecast.  Single-threaded; no mutex needed.
// ---------------------------------------------------------------------------
struct Cache {
    std::unique_ptr<ForecastGrid>                  grid;
    std::unique_ptr<interpolate::GridInterpolator> interp;

    bool has_grid() const noexcept { return grid != nullptr; }

    void set(ForecastGrid g) {
        grid   = std::make_unique<ForecastGrid>(std::move(g));
        interp = std::make_unique<interpolate::GridInterpolator>(*grid);
    }
};

// ---------------------------------------------------------------------------
// ServerState — groups mutable server state passed to handle_request().
// ---------------------------------------------------------------------------
struct ServerState {
    Cache cache;

#ifdef ROPE_HAS_FORECAST
    std::unique_ptr<forecast::Pipeline> pipeline;
#endif
};

// ---------------------------------------------------------------------------
// Request handler.
// Returns true  → keep this connection open.
// Returns false → close this connection (exit or fatal framing error).
// Sets exit_requested=true when the "exit" message is received.
// ---------------------------------------------------------------------------
static bool handle_request(platform::IpcSocket& sock,
                            ServerState&         state,
                            bool&                exit_requested) {
    json req  = wire::recv_msg(sock);
    auto type = req.value("type", "");

    try {
        // ----------------------------------------------------------------
        // forecast — run inference and cache the resulting grid.
        // ----------------------------------------------------------------
        if (type == "forecast") {
#ifndef ROPE_HAS_FORECAST
            wire::send_msg(sock,error_resp("no_forecast",
                "ROPE was built without ONNX Runtime; forecast is unavailable"));
            return true;
#else
            if (!state.pipeline) {
                wire::send_msg(sock,error_resp("no_forecast",
                    "forecast pipeline failed to load at startup; "
                    "check server logs"));
                return true;
            }

            std::string start_iso = req.at("start").get<std::string>();
            int horizon = req.value("horizon", 120);

            ForecastGrid grid = state.pipeline->run(start_iso, horizon);

            std::string ws = format_iso(grid.times.front());
            std::string we = format_iso(grid.times.back());

            state.cache.set(std::move(grid));

            wire::send_msg(sock,{
                {"status",       "ok"},
                {"window_start", ws},
                {"window_end",   we}
            });
            return true;
#endif
        }

        // ----------------------------------------------------------------
        // get — single-point interpolation query.
        // ----------------------------------------------------------------
        if (type == "get") {
            if (!state.cache.has_grid())
                throw std::runtime_error(
                    "no forecast cached; run 'rope forecast' first");

            std::string mode = req.at("mode").get<std::string>();
            TimePoint   tp   = parse_datetime(req.at("time").get<std::string>());
            double      lst  = req.at("lst").get<double>();
            double      lat  = req.at("lat").get<double>();
            double      alt  = req.at("alt").get<double>();

            auto r = (mode == "hold")
                ? state.cache.interp->query_hold(tp, lst, lat, alt)
                : state.cache.interp->query_interp(tp, lst, lat, alt);

            wire::send_msg(sock,{
                {"status",      "ok"},
                {"density",     r.density},
                {"uncertainty", r.uncertainty}
            });
            return true;
        }

        // ----------------------------------------------------------------
        // batch_get — multi-point interpolation query.
        // ----------------------------------------------------------------
        if (type == "batch_get") {
            if (!state.cache.has_grid())
                throw std::runtime_error(
                    "no forecast cached; run 'rope forecast' first");

            std::string mode    = req.at("mode").get<std::string>();
            const auto& pts     = req.at("points");
            json        results = json::array();

            for (const auto& p : pts) {
                TimePoint tp = parse_datetime(p.at("time").get<std::string>());
                double lst   = p.at("lst").get<double>();
                double lat   = p.at("lat").get<double>();
                double alt   = p.at("alt").get<double>();

                auto r = (mode == "hold")
                    ? state.cache.interp->query_hold(tp, lst, lat, alt)
                    : state.cache.interp->query_interp(tp, lst, lat, alt);

                results.push_back({
                    {"density",     r.density},
                    {"uncertainty", r.uncertainty}
                });
            }

            wire::send_msg(sock,{{"status", "ok"}, {"results", std::move(results)}});
            return true;
        }

        // ----------------------------------------------------------------
        // fetch_grid — send the full cached grid to the client.
        // ----------------------------------------------------------------
        if (type == "fetch_grid") {
            if (!state.cache.has_grid())
                throw std::runtime_error(
                    "no forecast cached; run 'rope forecast' first");

            const ForecastGrid& g = *state.cache.grid;

            json datetimes = json::array();
            for (auto t : g.times)
                datetimes.push_back(format_iso(t));

            wire::send_msg(sock,{
                {"status",       "ok"},
                {"H",            g.H},
                {"window_start", format_iso(g.times.front())},
                {"window_end",   format_iso(g.times.back())},
                {"datetimes",    std::move(datetimes)},
                {"density",      rope::detail::base64_encode(g.density)},
                {"uncertainty",  rope::detail::base64_encode(g.uncertainty)}
            });
            return true;
        }

        // ----------------------------------------------------------------
        // exit — clean shutdown.
        // ----------------------------------------------------------------
        if (type == "exit") {
            wire::send_msg(sock,{{"status", "ok"}});
            exit_requested = true;
            return false;
        }

        wire::send_msg(sock,error_resp("bad_request", "unknown type: " + type));
        return true;

    } catch (const interpolate::TimeOutOfRangeError& e) {
        wire::send_msg(sock,error_resp("time_out_of_range", e.what()));
        return true;
    } catch (const interpolate::SpatialOutOfRangeError& e) {
        wire::send_msg(sock,error_resp("spatial_out_of_range", e.what()));
        return true;
    } catch (const std::exception& e) {
        wire::send_msg(sock,error_resp("internal", e.what()));
        return true;
    }
}

// ---------------------------------------------------------------------------
// run — entry point called by cli/main.cpp via --serve.
// ---------------------------------------------------------------------------
void run(const std::filesystem::path& socket_path,
         const std::filesystem::path& config_path) {
    // Install signal handlers.
    g_running = true;
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    // Validate config early so missing keys fail before we bind the socket.
    io::ConfigReader config{config_path};

    // Crash recovery: probe the socket.
    {
        bool already_running = false;
        try {
            auto probe = platform::IpcSocket::connect(socket_path);
            already_running = true;
        } catch (...) {}

        if (already_running)
            throw std::runtime_error(
                "rope server: another instance is already running at " +
                socket_path.string());

        std::error_code ec;
        std::filesystem::remove(socket_path, ec);
        std::filesystem::create_directories(socket_path.parent_path(), ec);
    }

    // Bind and listen.
    auto srv = platform::ServerSocket::bind(socket_path);

    // Load the forecast pipeline from config.
    ServerState state;

#ifdef ROPE_HAS_FORECAST
    try {
        // Paths in the config file may be relative; resolve them against the
        // directory that contains the config file.
        const auto config_dir = config_path.parent_path();
        auto resolve = [&](const std::string& p) -> std::filesystem::path {
            std::filesystem::path fp{p};
            return fp.is_absolute() ? fp : config_dir / fp;
        };

        forecast::Config fcfg;
        fcfg.exported_dir        = resolve(config.get("paths.exported_dir"));
        // driver_path is optional: when absent, the cache manager takes over.
        if (config.has("paths.driver_path"))
            fcfg.driver_path = resolve(config.get("paths.driver_path"));
        // cache_dir is optional: defaults to platform cache root.
        if (config.has("driver_cache.cache_dir"))
            fcfg.cache_dir = resolve(config.get("driver_cache.cache_dir"));
        fcfg.cache_max_age_hours   = config.get_int("driver_cache.max_age_hours", 24);
        fcfg.intra_threads_base    = config.get_int("threads.intra_threads_base", 1);
        fcfg.intra_threads_meta    = config.get_int("threads.intra_threads_meta", 0);
        fcfg.intra_threads_decoder = config.get_int("threads.intra_threads_decoder", 0);
        fcfg.decoder_device        = config.get("decoder.device", "cpu");
        fcfg.compute_uncertainty   = config.get("forecast.compute_uncertainty", "true") == "true";

        state.pipeline = forecast::load(fcfg);
    } catch (const std::exception& e) {
        // Log the error but keep running — the server can still serve
        // cached grids from a previous session via fetch_grid.
        // forecast requests will return a clear error message.
        std::fprintf(stderr,
            "rope server: forecast pipeline load failed: %s\n"
            "  forecast requests will return an error.\n", e.what());
    }
#endif

    // Idle timeout.
    using Clock = std::chrono::steady_clock;
    using Ms    = std::chrono::milliseconds;
    auto now_ms = []() -> std::int64_t {
        return std::chrono::duration_cast<Ms>(Clock::now().time_since_epoch()).count();
    };

    const int idle_timeout_seconds = config.get_int("server.idle_timeout_seconds", 1800);
    std::atomic<std::int64_t> last_activity_ms{now_ms()};

    std::thread watcher;
    if (idle_timeout_seconds > 0) {
        const std::int64_t timeout_ms = static_cast<std::int64_t>(idle_timeout_seconds) * 1000;
        watcher = std::thread([&last_activity_ms, timeout_ms, &now_ms]() {
            while (g_running) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                if (!g_running) break;
                if (now_ms() - last_activity_ms.load() >= timeout_ms)
                    g_running = false;
            }
        });
    }

    // Accept loop.
    bool exit_requested = false;

    while (g_running && !exit_requested) {
        auto conn = srv->accept(g_running);
        if (!conn) break;

        bool keep_conn = true;
        while (keep_conn && g_running && !exit_requested) {
            try {
                keep_conn = handle_request(*conn, state, exit_requested);
                if (keep_conn) last_activity_ms.store(now_ms());
            } catch (const std::exception&) {
                keep_conn = false;
            }
        }
    }

    if (watcher.joinable())
        watcher.join();
}

} // namespace rope::server
