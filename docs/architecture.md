# Architecture

ROPE is an **inference-only** atmospheric density forecasting service. This codebase loads pre-trained model artifacts, runs inference on demand, writes the resulting grid to a cache file, and serves interpolation queries against it.

---

## System overview

```
  User / script
       │
       │  CLI commands  (rope forecast / get)
       ▼
  ┌─────────┐
  │   CLI   │
  │ src/cli │
  └────┬────┘
       │
       │  forecast: runs the pipeline in-process, then
       │            atomically writes the cache file
       │  get:      memory-maps the cache file and interpolates
       ▼
  ┌─────────────────────────────┐        ┌─────────┐
  │   Forecast-grid cache file  │◄──mmap─│  C API  │  (rope_open, in another process)
  │   (ForecastGridBin /        │        │src/capi │
  │    MappedForecastGrid)      │        └─────────┘
  └──────────────┬───────────────┘
                 │
      ┌──────────▼──────────┐
      │    Interpolator      │
      │  src/interpolate/     │
      │  GridInterpolator<Grid> │
      └───────────────────────┘
```

There is no background process and no IPC. `rope forecast` and `rope get` are ordinary, short-lived CLI invocations; the C API (`rope_open`) reads the same cache file directly, typically from a different process (e.g. a Python script via ctypes) that opens it once and issues many queries against the in-process mapping.

---

## Module map

| Directory | Namespace | Role |
|-----------|-----------|------|
| `src/cli/` | (binary entry point) | Argument parsing, running forecasts, querying the cache file, output formatting |
| `src/forecast/` | `rope::forecast` | The full inference pipeline: driver loading through density decoding |
| `src/interpolate/` | `rope::interpolate` | Grid interpolation; `GridInterpolator<Grid>` works over either the in-memory or memory-mapped grid representation |
| `src/io/` | `rope::io` | File I/O: CSV/binary readers, config, stats, driver DB, IC table, forecast-grid cache file (`ForecastGridBin`, `MappedForecastGrid`) |
| `src/capi/` | (C ABI) | Thin shared-library wrapper exposing `rope_open` / `rope_query` — memory-maps the cache file, no ONNX Runtime/libtorch dependency |
| `src/core/platform/` | `rope::platform` | OS-specific: path resolution, memory-mapped files, exe path |

Public headers mirror the source layout under `include/rope/<module>/`. Nothing in `src/` is visible outside its own module except through the declared public header.

---

## Data flow

```
rope.conf
    └─ paths.exported_dir → model artifacts (ONNX, .bin, JSON)
    └─ paths.driver_path  → sw_celestrack.swbin (or cache manager)

At forecast time (`rope forecast`, runs once per invocation):
  1. SpaceWeatherDB         reads  .swbin → hourly F10.7, Kp, harmonics
  2. ICTable                reads  ic_table.icbin → (F10, Kp) → K-D latent IC
  3. (pipeline internals)   builds X_init (S, D) and x_chunk (H+1, S, D)
  4. SlidingWindowRollout   runs M base models → base_latents (M, H, K)
  5. EnsembleFuser          fuses base_latents → mu_lat (H+1, K) + spread
  6. UT (optional)          propagates uncertainty through 2K+1 sigma points
  7. LatentDecoder          decodes mu_lat → density (H, n_lst, n_lat, n_alt) per this model's GridSpec
  8. ForecastGrid           stores density + uncertainty
  9. ForecastGridBin::save  atomically writes the grid to the cache file, discarding any previous one

At query time (fast path), from `rope get` or the C API:
  MappedForecastGrid::open  memory-maps the cache file (or ForecastGridBin::load for a fully in-memory copy)
  GridInterpolator<Grid>.query_interp(time, lst, lat, alt_km)
      → trilinear spatial + temporal linear interpolation in log10 space
      → (density, uncertainty)
```

---

## Key design principles

**Inference only.** No training, no fine-tuning, no weight updates. The service is stateless across forecasts from the model's perspective.

**Cache-first.** Inference is the slow path (seconds). Interpolation is the fast path (microseconds). The grid is computed once per `rope forecast` call and written to a cache file, then re-used for all subsequent queries until the next forecast.

**Uncertainty is mandatory.** Every query returns density AND uncertainty. The Unscented Transform propagates ensemble spread through the decoder to produce a physically grounded uncertainty estimate. Skipping uncertainty halves decoder calls but zeroes the uncertainty field — the result is never returned without it.

**Fail loudly.** Missing files, bad magic numbers, out-of-range queries, version mismatches — all throw with a clear message. There is no silent fallback or clamping anywhere in the inference stack.

**Single cache slot, last-write-wins.** There is exactly one forecast-grid cache file per user. A new `rope forecast` atomically replaces it (write to a temp file, then rename) — a reader never observes a partial file, and a failed forecast never touches an existing good cache. There is no server process coordinating this; the filesystem is the only shared state.

**Per-model grid shape.** LST/lat/alt bin counts and physical ranges (`GridSpec`) are declared per model in `model_manifest.json`, not a fixed global constant. The cache file's header carries the full `GridSpec` (not just counts) since a reader never has access to the manifest that produced it.

**No OS-specific code outside `platform/`.** The Linux, macOS, and Windows implementations of path resolution and memory-mapped files live only in `src/core/platform/posix.cpp` and `src/core/platform/windows.cpp`. All other source files compile identically on all three platforms.

---

## Dependency graph (static libraries)

```
rope_exe  →  rope_io          →  rope_core
          →  rope_interpolate →  rope_io  →  rope_core
          →  rope_forecast    →  rope_io
                               →  ORT / LibTorch

rope.so   →  rope_io          →  rope_core
          →  rope_interpolate
```

`rope_forecast` links against ONNX Runtime (always) and LibTorch (when `ROPE_USE_LIBTORCH=ON`) and is linked only into the `rope_exe` CLI binary — everything else, including the C API shared library, has no ML runtime dependency.

---

## Related documents

- [building.md](building.md) — how to compile and test
- [pipeline.md](pipeline.md) — the forecast pipeline in detail
- [driver-system.md](driver-system.md) — driver data, cache, binary formats
- [interpolation.md](interpolation.md) — grid structure and query logic
- [forecast-cache-format.md](forecast-cache-format.md) — the on-disk cache file `rope forecast` writes and `rope get`/the C API read
- [model-artifacts.md](model-artifacts.md) — ONNX models, stats files, JSON configs
