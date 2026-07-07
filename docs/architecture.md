# Architecture

ROPE is an **inference-only** atmospheric density forecasting service. This codebase loads pre-trained model artifacts, runs inference on demand, caches the resulting grid, and serves interpolation queries against it.

---

## System overview

```
  User / script
       │
       │  CLI commands  (rope forecast / get / exit)
       ▼
  ┌─────────┐  IPC socket ┌───────────────────────────────┐
  │   CLI   │◄───────────►│            Server             │
  │ src/cli │             │        src/server/            │
  └────┬────┘             │                               │
       │                  │  ┌──────────┐  ┌───────────┐  │
       │ P/Invoke         │  │ Pipeline │  │   Cache   │  │
  ┌────▼────┐             │  │src/fore- │  │(ForecastG-│  │
  │  C API  │  fetch_grid │  │  cast/   │  │   rid)    │  │
  │src/capi │◄────────────┤  └──────────┘  └─────┬─────┘  │
  └─────────┘             │                      │        │
                          │  ┌───────────────────▼──────┐ │
                          │  │      Interpolator        │ │
                          │  │   src/interpolate/       │ │
                          │  └──────────────────────────┘ │
                          └───────────────────────────────┘
```

There is exactly one server per user, identified by a per-user socket path. All interaction—whether from the CLI, a Python script using the C API, or a C# application—ultimately goes through this server or the in-process interpolation handle loaded from it.

---

## Module map

| Directory | Namespace | Role |
|-----------|-----------|------|
| `src/cli/` | (binary entry point) | Argument parsing, server lifecycle, output formatting |
| `src/client/` | `rope::client` | IPC socket transport; typed request/response API used by CLI |
| `src/server/` | `rope::server` | Long-lived server: accepts requests, runs pipeline, caches grid |
| `src/forecast/` | `rope::forecast` | The full inference pipeline: driver loading through density decoding |
| `src/interpolate/` | `rope::interpolate` | In-process 4-D grid interpolation |
| `src/io/` | `rope::io` | File I/O: CSV/binary readers, config, stats, driver DB, IC table |
| `src/capi/` | (C ABI) | Thin shared-library wrapper exposing `rope_open` / `rope_query` |
| `src/core/platform/` | `rope::platform` | OS-specific: sockets, process spawn, exe path, cache directory |

Public headers mirror the source layout under `include/rope/<module>/`. Nothing in `src/` is visible outside its own module except through the declared public header.

---

## Data flow

```
rope.conf
    └─ paths.exported_dir → model artifacts (ONNX, .bin, JSON)
    └─ paths.driver_path  → sw_celestrack.swbin (or cache manager)

At forecast time:
  1. SpaceWeatherDB         reads  .swbin → hourly F10.7, Kp, harmonics
  2. ICTable                reads  ic_table.icbin → (F10, Kp) → K-D latent IC
  3. (pipeline internals)   builds X_init (S, D) and x_chunk (H+1, S, D)
  4. SlidingWindowRollout   runs M base models → base_latents (M, H, K)
  5. EnsembleFuser          fuses base_latents → mu_lat (H+1, K) + spread
  6. UT (optional)          propagates uncertainty through 2K+1 sigma points
  7. LatentDecoder          decodes mu_lat → density (H, 72, 36, 45)
  8. ForecastGrid           stores density + uncertainty; server caches it

At query time (fast path):
  GridInterpolator.query_interp(time, lst, lat, alt_km)
      → trilinear spatial + temporal linear interpolation in log10 space
      → (density, uncertainty)
```

---

## Key design principles

**Inference only.** No training, no fine-tuning, no weight updates. The service is stateless across forecasts from the model's perspective.

**Cache-first.** Inference is the slow path (seconds). Interpolation is the fast path (microseconds). The grid is computed once per `rope forecast` call and then re-used for all subsequent queries until the next forecast.

**Uncertainty is mandatory.** Every query returns density AND uncertainty. The Unscented Transform propagates ensemble spread through the decoder to produce a physically grounded uncertainty estimate. Skipping uncertainty halves decoder calls but zeroes the uncertainty field — the result is never returned without it.

**Fail loudly.** Missing files, bad magic numbers, out-of-range queries, version mismatches — all throw with a clear message. There is no silent fallback or clamping anywhere in the inference stack.

**One server per user.** The socket path is user-scoped. A second `rope forecast` from the same user connects to the already-running server rather than spawning a new one. The source of truth for "is there a server?" is the socket file: if nothing is listening, it's treated as absent even if the file exists.

**No OS-specific code outside `platform/`.** The Linux, macOS, and Windows implementations of sockets, process spawning, and path resolution live only in `src/core/platform/posix.cpp` and `src/core/platform/windows.cpp`. All other source files compile identically on all three platforms.

---

## Dependency graph (static libraries)

```
rope_exe  →  rope_client
          →  rope_server  →  rope_forecast  →  rope_io
                                            →  ORT / LibTorch
                          →  rope_interpolate
                          →  rope_io
          →  rope_io
          →  rope_core    →  platform layer

rope.so   →  rope_client
          →  rope_interpolate
          →  rope_core
```

`rope_forecast` links against ONNX Runtime (always) and LibTorch (when `ROPE_USE_LIBTORCH=ON`). Everything else is standard C++20 with no ML runtime dependency.

---

## Related documents

- [building.md](building.md) — how to compile and test
- [pipeline.md](pipeline.md) — the forecast pipeline in detail
- [driver-system.md](driver-system.md) — driver data, cache, binary formats
- [interpolation.md](interpolation.md) — 4-D grid structure and query logic
- [server-protocol.md](server-protocol.md) — IPC transport and server lifecycle
- [model-artifacts.md](model-artifacts.md) — ONNX models, stats files, JSON configs
