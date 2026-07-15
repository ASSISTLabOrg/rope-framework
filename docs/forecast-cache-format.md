# Forecast-Grid Cache File Format

The single artifact `rope forecast` produces and `rope get`/the C API consume. Replaces the old server/socket transport — there is no wire protocol anymore, just this on-disk file.

`include/rope/io/forecast_grid_bin.h`, `src/io/forecast_grid_bin.cpp` (write path, and a full-materializing read path used mainly by tests); `include/rope/io/mapped_forecast_grid.h`, `src/io/mapped_forecast_grid.cpp` (the production, memory-mapped read path used by `rope get` and `rope_open`).

## Location

Exactly one cache file per user (see `cli.md`):

- **Linux/macOS:** `$XDG_CACHE_HOME/rope/forecast_grid.bin`, fallback `~/.cache/rope/forecast_grid.bin`
- **Windows:** `%LOCALAPPDATA%\rope\forecast_grid.bin`

## Layout

Little-endian throughout. Flat and fixed-stride — no index — so any timestep's byte offset is directly computable, which is what makes memory-mapped reads possible without parsing the whole file.

```
Header (60 bytes):
  uint32  magic       = 0x52504647  ("RPFG")
  uint32  version     = 1
  int32   n_lst, n_lat, n_alt   -- this forecast's GridSpec bin counts
  int32   H                     -- forecast hours
  double  lat_min_deg, lat_max_deg
  double  alt_min_km,  alt_max_km
  uint32  reserved    = 0

Body:
  int64   times[H]                                    -- UTC seconds since epoch
  float32 density[H * n_lst*n_lat*n_alt]              -- kg/m³, row-major [t, lst, lat, alt]
  float32 uncertainty[H * n_lst*n_lat*n_alt]           -- kg/m³, same layout
```

The header carries the *full* `GridSpec` — counts **and** physical ranges — not just counts. A process reading this file (`rope get`, `rope_open`) never has access to the `model_manifest.json` that produced it, so the cache file must be fully self-describing. `GridSpec` is declared per model in `model_manifest.json` (documented in the project's agent-facing knowledge base as `model-registry.md`); see `interpolation.md` for what these fields mean physically.

Header fields are read/written individually (not via a single packed-struct read) to avoid compiler-dependent padding between the mixed `int32`/`double` fields.

## Validation

`ForecastGridBin::load()`/`MappedForecastGrid::open()` check, in order: file exists (else `ForecastCacheMissingError`), magic matches, version is supported, `n_lst/n_lat/n_alt`/`H` are positive and sane, `lat_min_deg < lat_max_deg` and `alt_min_km < alt_max_km`, and the file is at least as large as the header + body size implied by its own `H`/shape fields — any violation throws `ForecastCacheCorruptError` (`include/rope/io/forecast_cache_errors.h`). Consistent with `forecast-invariants.md`: a malformed cache fails loudly, it is never silently reinterpreted or clamped.

## Writing — atomicity

`ForecastGridBin::save()` writes to a temporary file in the *same directory* as the target (never a system temp directory — cross-filesystem renames can silently degrade to non-atomic copy+delete on some standard library implementations), then calls `std::filesystem::rename()` to replace the real cache path. A reader never observes a partially-written file: the target path always holds either the previous complete forecast or the new one, never a mix. If `save()` fails partway through, the temp file is removed and the existing cache file (if any) is untouched.

This is also what gives the "second forecast discards the first" invariant: there is exactly one cache path, and each successful `rope forecast` atomically replaces whatever was there before.

## Reading — memory-mapped snapshot semantics

`MappedForecastGrid::open()` memory-maps the file read-only (`rope::platform::MappedFile`) rather than copying `density`/`uncertainty` into owned buffers — the OS pages in only the bytes a query actually touches, which is what keeps multi-year (tens-of-GB) forecasts practical to query. `H`, the `GridSpec`, and `times` (small — at most a few hundred KB even for a multi-year hourly forecast) are copied into ordinary owned fields at open time.

On POSIX, if a concurrent `rope forecast` atomically replaces the file while a reader still has the old one mapped, the mapping keeps serving the old bytes until it's released — `rename`/`unlink` never invalidate an existing mapping or file descriptor. The Windows implementation opens with `FILE_SHARE_DELETE` (alongside `FILE_SHARE_READ`) to reproduce the same property, since Windows does not give it for free otherwise.

## Related

- Grid semantics and physical layout: `interpolation.md`
- C API (`rope_open`, error codes) and CLI commands: documented in the project's agent-facing knowledge base (`capi.md`, `cli.md`), not duplicated in this tree
- Implementation: `src/io/forecast_grid_bin.*`, `src/io/mapped_forecast_grid.*`, `src/core/platform/{posix,windows}.cpp` (`MappedFile`)
