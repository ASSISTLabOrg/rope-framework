# I/O Components

All file I/O lives under `src/io/` with public headers in `include/rope/io/`. This module has no dependency on the ML runtime.

---

## CsvReader

`include/rope/io/csv_reader.h`, `src/io/csv_reader.cpp`

In-memory column-oriented CSV reader. Loads the entire file at construction time.

```cpp
CsvReader csv(path);
csv.nrows()                    // row count
csv.has_column("name")         // column existence check
csv.get("name", row)           // string value
csv.get_float("name", row)     // float value
csv.get_int("name", row)       // int value
```

Handles quoted fields, leading/trailing whitespace, and case-insensitive column lookup (after stripping whitespace). Used by `SpaceWeatherDB` (CSV path), `ICTable` (CSV path), and batch `rope get --file`.

---

## ConfigReader

`include/rope/io/config_reader.h`, `src/io/config_reader.cpp`

INI-style key-value reader. Sections are written as `[section]`; keys are accessed as `"section.key"`. Comments start with `#`.

```cpp
ConfigReader cfg(path);
cfg.has("paths.driver_path")                   // existence check
cfg.get("paths.driver_path")                   // required; throws if absent
cfg.get("decoder.device", "cpu")               // optional with default
cfg.get_int("threads.intra_threads_base", 1)
cfg.get_double("some.float", 0.0)
```

Relative paths in the config are resolved by the caller relative to the config file's own directory — the `ConfigReader` itself does not resolve paths.

---

## Stats — `stats.h`

`include/rope/io/stats.h` (header-only)

**`Stats`** — raw statistics blob loaded from a `.bin` file:

```cpp
Stats s = Stats::load(path);
// s.shape, s.mu, s.sigma
```

Binary format: `uint32 ndim`, `uint32 shape[ndim]`, `float32 mu[N]`, `float32 sigma[N]`.

**`FeatureNormalizer`** — wraps a `Stats` for the time-series feature vector `[latent | driver]`:

```cpp
FeatureNormalizer norm(stats_ts, K);
norm.latent_dim()    // K
norm.driver_dim()    // D - K
norm.total_dim()     // D

norm.norm_full_inplace(float* x)      // normalize (K + driver_dim) vector in-place
norm.norm_driver_inplace(float* drv)  // normalize driver portion only
norm.denorm_latents_inplace(float* z) // recover physical latent from normalized
norm.denorm_latents_block(float* z, int rows) // batch version
```

**`CAEDenormalizer`** — wraps a `Stats` for the decoder output:

```cpp
CAEDenormalizer dn(stats_cae);
dn.apply_inplace(float* data, int batch, int voxels_with_ch);
// data[i] = 10^(data[i] * sigma[i] + mu[i])
```

Supports both scalar and per-voxel stats.

---

## Driver I/O

### SpaceWeatherDB

See [driver-system.md](driver-system.md) for the full description. Summary:

```cpp
// Load from file (extension determines format)
auto db = SpaceWeatherDB::from_file(path);  // .swbin or CSV

// Point lookup (throws if not found)
DriverRow row = db.lookup(tp);

// Window builder
auto rows = DriverWindowBuilder::build(db, start_iso, H+1, S);
```

### SpaceWeatherBin

```cpp
auto db  = SpaceWeatherBin::load(bin_path);  // binary → SpaceWeatherDB
SpaceWeatherBin::save(db, bin_path);          // SpaceWeatherDB → binary
```

### DriverCacheManager

```cpp
DriverCacheManager mgr(cache_dir, max_age_hours);
fs::path path = mgr.get_path("celestrak_sw");  // → fresh .swbin path
```

---

## IC table I/O

### ICTable

```cpp
// Load from file (extension determines format)
auto table = ICTable::from_file(path);  // .icbin or CSV
// or directly:
ICTable table(csv_path);

auto coeffs = table.get_latent_coeffs(f10, kp);  // std::vector<float>(K)
table.latent_dim()  // K
```

### IcBin

```cpp
auto table = IcBin::load(bin_path);    // binary → ICTable
IcBin::save(table, bin_path);          // ICTable → binary
```

`grid_axes`/`latent_dim` come from `model_manifest.json` directly (`ModelManifest::ic_grid_axes`/`::latent_dim`) — no separate config file. See [driver-system.md](driver-system.md).

---

## Binary format details

### Magic number validation

Both `.swbin` and `.icbin` validate a magic number and version on load and throw `std::runtime_error` on mismatch. This prevents silent corruption from incorrect file types or git line-ending conversion.

| Format | Magic | Hex |
|--------|-------|-----|
| `.swbin` | `RPSW` | `0x52505357` |
| `.icbin` | `RPIC` | `0x52504943` |

All multi-byte fields are little-endian. Fixed-width integer types (`int64`, `uint32`, `float32`, `int32`) are used throughout to guarantee cross-platform byte layout.

### Endianness note

The project targets x86_64 (little-endian) on all platforms. The binary formats are explicitly little-endian by design. If the project is ever ported to a big-endian architecture, the binary loaders would need byte-swap logic.

---

## CLI convert subcommands

```
rope convert-sw --input <csv_or_swbin> --output <swbin>
rope convert-ic --input <csv_or_icbin> --output <icbin>
```

Both accept either format as input (`from_file()` dispatches on extension) and write the binary output. Implemented in `src/cli/main.cpp`.

---

## Adding a new file format

To add a new binary format:
1. Define the magic constant and format spec
2. Create `src/io/<name>_bin.cpp` and `include/rope/io/<name>_bin.h` with `load()` and `save()`
3. Add a private constructor to the target class and `friend class <Name>Bin`
4. Add `from_file()` to the target class dispatching on extension
5. Add the new source to the `rope_io` target in `CMakeLists.txt`
6. Add a CLI subcommand in `src/cli/main.cpp` if user-facing conversion is needed
