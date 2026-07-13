# Server and IPC Protocol

---

## Server overview

The server is the same `rope` binary invoked with a hidden `--serve --socket-path <p> --config-path <p>` flag. It is spawned as a detached background process by the CLI and is never started by users directly.

**Key properties:**
- One server per user, identified by a per-user socket path
- Long-lived: stays running between `rope forecast` calls
- Synchronous: one request is processed at a time (no concurrency inside the server)
- Cache-first: holds the `ForecastGrid` in memory between requests

---

## Socket path (per-user)

| Platform | Default path |
|----------|-------------|
| Linux/macOS | `$XDG_RUNTIME_DIR/rope.sock`, fallback `/tmp/rope-$UID.sock` |
| Windows | `%LOCALAPPDATA%\rope\rope.sock` |

Resolved in `src/core/platform/posix.cpp::default_socket_path()` and `windows.cpp::default_socket_path()`. Override with `--socket <path>` in any CLI command.

---

## Server lifecycle

### Startup (in `server::run()`)

1. Read and validate `rope.conf` early (fail before binding)
2. **Crash recovery:** attempt to connect to the existing socket. If something is listening → throw "another instance is already running". If not → remove stale socket file and proceed.
3. Bind the socket and start listening
4. Load the forecast pipeline (`forecast::load(fcfg)`) — this is the slow step (loading 15+ ONNX models). If it fails, the server logs to stderr and continues with `pipeline = nullptr`. Forecast requests then return a clear error.
5. Enter the accept loop

### Idle timeout

A watcher thread checks `last_activity_ms` every second. If no request has been processed for `idle_timeout_seconds` (default 1800, set to 0 to disable), it sets `g_running = false`, which causes the accept loop to exit and the server to shut down cleanly.

### Shutdown

`SIGINT` / `SIGTERM` (POSIX) or an `exit` request sets `g_running = false`. The accept loop checks this flag on each timeout and exits cleanly after the current request (if any) finishes.

---

## Wire format

**Length-prefixed JSON.** Each message is:
```
[4 bytes: uint32 little-endian length] [length bytes: UTF-8 JSON]
```

Strictly synchronous: the client sends one request and waits for one response. There are no server-initiated messages.

---

## Request types

### `forecast`

Runs inference and caches the resulting grid.

```json
{ "type": "forecast", "start": "2024-06-01T00:00:00", "horizon": 24 }
```
```json
{ "status": "ok", "window_start": "2024-06-01T01:00:00", "window_end": "2024-06-02T00:00:00" }
```

Note: `window_start` is `start + 1h` (the first genuine prediction) and `window_end` is `start + H`.

### `get`

Single-point interpolation query against the cached grid.

```json
{ "type": "get", "mode": "interp", "time": "2024-06-01T07:00:00", "lst": 12.5, "lat": 45.0, "alt": 400.0 }
```
```json
{ "status": "ok", "density": 4.72e-12, "uncertainty": 3.5e-13 }
```

`mode`: `"hold"` or `"interp"`.

### `batch_get`

N-point interpolation in one round-trip. Results are in the same order as input.

```json
{
  "type": "batch_get", "mode": "interp",
  "points": [
    { "time": "2024-06-01T07:00:00", "lst": 12.5, "lat": 45.0, "alt": 400.0 },
    { "time": "2024-06-01T08:00:00", "lst": 6.0,  "lat": -30.0, "alt": 300.0 }
  ]
}
```
```json
{ "status": "ok", "results": [{ "density": 4.72e-12, "uncertainty": 3.5e-13 }, ...] }
```

### `fetch_grid`

Downloads the full cached grid as Base64-encoded binary blobs. Used by the C API during `rope_open` so that subsequent queries run in-process with zero socket overhead.

```json
{ "type": "fetch_grid" }
```
```json
{
  "status": "ok",
  "H": 24,
  "window_start": "...",
  "window_end": "...",
  "datetimes": ["2024-06-01T01:00:00", "..."],
  "density":     "<base64: H × 116640 float32 LE>",
  "uncertainty": "<base64: H × 116640 float32 LE>"
}
```

Grid layout: row-major `[t, lst, lat, alt]`. See [interpolation.md](interpolation.md).

### `exit`

```json
{ "type": "exit" }
```
```json
{ "status": "ok" }
```

### Error response

Returned for any request type on failure:
```json
{ "status": "error", "code": "time_out_of_range", "message": "..." }
```

Error codes: `time_out_of_range`, `spatial_out_of_range`, `bad_request`, `no_forecast`, `no_forecast` (pipeline failed at startup), `internal`.

---

## Server cache structure

```cpp
struct Cache {
    std::unique_ptr<ForecastGrid>                  grid;
    std::unique_ptr<interpolate::GridInterpolator> interp;

    bool has_grid() const noexcept;
    void set(ForecastGrid g);  // replaces both grid and interp atomically
};
```

When a new forecast completes, `cache.set(std::move(grid))` rebuilds the interpolator atomically. There is no locking — the server is single-threaded.

---

## Server spawn mechanism

Implemented in `src/core/platform/`:

**POSIX (`posix.cpp`):** `posix_spawn` with `POSIX_SPAWN_SETPGROUP` so the child survives the parent exiting. stdin/stdout/stderr are redirected to `/dev/null` so the server does not inherit captured pipes from the calling process.

**Windows (`windows.cpp`):** `CreateProcessA` with `CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS`. `lpEnvironment = NULL` so the child inherits the parent's environment (including PATH, which must include the ORT DLL directory).

The CLI polls the socket for up to 6 seconds (200 ms intervals) to detect when the server is ready. If the socket never appears, it prints "timed out waiting for server to start" and exits with code 1.

---

## CLI → client → server boundary rules

- `cli` spawns the server and manages its lifecycle, but uses `client::IpcClient` for all IPC
- `cli` must not include `src/server/` headers
- `server` must not include `src/cli/` or `src/client/` headers
- Types crossing the socket are serialized as JSON; no internal C++ types leak across the boundary
