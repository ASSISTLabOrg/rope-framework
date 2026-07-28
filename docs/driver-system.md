# Driver System

The driver system loads hourly space-weather data (commonly F10.7 solar flux and Kp geomagnetic index, but not limited to them — see below) and makes it available to the pipeline as a sequence of `DriverRow` feature vectors. It also manages the IC table, which maps a 2-axis grid (again, commonly F10/Kp) to initial latent conditions.

**Design principle:** a model's driver set — which raw columns it needs, how many, what they're named — is declared entirely by its manifest, not hardcoded anywhere in this system. Core types carry no references to specific driver identities; only a per-model manifest, a specific named CSV file, or a specific named online source ever says "f10" or "kp" literally.

---

## DriverRow

Defined in `include/rope/io/driver_db.h`. Represents one hour of driver data:

| Field | Meaning |
|-------|---------|
| `tp` | UTC timestamp (seconds since Unix epoch) |
| `t1` | `sin(2π·hour/24)` — diurnal harmonic |
| `t2` | `cos(2π·hour/24)` — diurnal harmonic |
| `t3` | `sin(2π·doy/365.25)` — annual harmonic |
| `t4` | `cos(2π·doy/365.25)` — annual harmonic |
| `raw` | `vector<pair<string,float>>` — whatever named columns the source (CSV or `.swbin`) actually provided |

`t1`–`t4` are **always derived** from `tp` via `harmonics()` in `datetime.h`, every time a row is built — a raw source that defines a column literally named `t1`/`t2`/`t3`/`t4` is rejected at load time (reserved names; never silently shadowed).

`DriverRow::get(name)` is the one place that resolves a driver name to a value:
1. `t1`–`t4` → the fields above, unconditionally.
2. Otherwise, linear-scan `raw` for `name`.
3. If still unresolved and `name` is `doy` or `hour_int` → derived from `tp` as a fallback (these two are "raw-if-present, else-derived" — read from the source when the source has them, computed otherwise; this is the one existing behavior this generalization preserves exactly).
4. Otherwise → throws. Never substitutes a value.

`f10`/`kp` are **not** privileged in any way at this layer — they're just whatever names happen to be in a given model's raw data. A future model with an entirely different, differently-sized driver set (no overlap with `f10`/`kp` at all) works the same way.

---

## SpaceWeatherDB

`include/rope/io/driver_db.h`, `src/io/driver_db.cpp`

An in-memory column-oriented store of hourly space-weather data. Loads when `rope forecast` constructs the pipeline; never modified at runtime.

**Internal layout:**
```
times_     : vector<TimePoint>          // sorted; binary-search index
raw_names_ : vector<string>             // one entry per raw column
raw_data_  : vector<vector<float>>      // raw_data_[j][i] = raw_names_[j]'s value at row i
```

**Loading:**
- `SpaceWeatherDB(csv_path)` — parses a CSV with a required `datetime` column plus *any* other columns present (via `CsvReader::column_names()`, which is already name-keyed and order-independent — column order in the file never matters). Rejects `t1`/`t2`/`t3`/`t4` as literal headers (reserved). `doy`/`hour_int` are backfilled from the timestamp only when genuinely absent from the file.
- `SpaceWeatherBin::load(bin_path)` → `SpaceWeatherDB` — reads `.swbin` (v2, self-describing) directly into the same generic column store.
- `SpaceWeatherDB::from_file(path)` — dispatches on `.swbin` extension vs. CSV.

**Lookup:** `lookup(tp)` runs `std::lower_bound` on `times_` (O(log N), unaffected by any of the above), then calls `make_row()` which assembles a `DriverRow` and recomputes harmonics.

**DriverWindowBuilder:** builds the `(S-1)+(H+1)` contiguous hourly rows needed by the pipeline. Throws if any hour is missing — no gap-filling.

---

## Driver columns and source

Declared in `model_manifest.json`'s top-level `drivers` block (required, kind-agnostic — sibling to `ic`/`grid`, not nested inside any model kind's own block, since driver sourcing is a cross-kind concern like model-backend selection or IC sourcing):

```jsonc
"drivers": {
  "source": "celestrak_sw",
  "columns": [
    { "name": "f10", "description": "F10.7 cm solar radio flux (SFU)." },
    { "name": "kp",  "description": "Kp planetary geomagnetic index (0-9 scale)." },
    { "name": "t1",  "description": "sin(2*pi*hour/24) - diurnal phase harmonic." }
  ]
}
```

`columns` is an ordered array (order = the array's position = the model's feature-vector order), each entry a self-contained `{name, description}` pair — the description is a copy written at export time, not a live reference into any registry. Parsed by `ModelManifest::load()` into `driver_columns` (names only, in order) and `driver_column_info` (name + description). `source` is a registered online source name (see `DriverCacheManager` below).

A canonical registry of known driver names — `driver_registry.json` in the `rope-registry` repository — lists every known name with `{name, kind: "raw"|"derived", description}`. It's metadata only: consulted by manifest-producing tooling (to default-fill descriptions and catch typos) and by a drift-detection test (`tests/cpp/test_driver_registry.cpp`, checking that `known_derived_driver_names()` matches the registry's `"derived"` entries) — never consulted by this codebase at inference time.

---

## DriverCacheManager

`include/rope/io/driver_cache.h`, `src/io/driver_cache.cpp`

Manages a local cache of `.swbin` files refreshed from online sources. Integrated into `Pipeline::load()` so auto-refresh fires for every entry point (CLI, C API, Python, C#).

**Source registry** (add entries in `driver_cache.cpp`):

| Key | URL |
|-----|-----|
| `celestrak_sw` | `https://celestrak.org/SpaceData/SW-Last5Years.csv` |
| `celestrak_sw_all` | `https://celestrak.org/SpaceData/SW-All.csv` |

Each named source is specific to its own known raw-column shape (CelesTrak only ever produces `f10`/`kp`) — that's a property of the *source*, not an assumption baked into core infrastructure; a model with a different driver set simply can't use `drivers.source: "celestrak_sw"` and needs an explicit `driver_path`/`--driver` override instead (see below).

**`get_path(source)`** returns the path to a fresh `.swbin` file:
1. If the cached file is absent or older than `max_age_hours` → calls `refresh(source, dest)`
2. If refresh fails but a stale file exists → logs a warning and returns the stale file
3. If no file at all → throws

**`refresh(source, dest)`:**
1. `download(url)` — HTTP GET (not yet implemented; throws with a clear message pointing to `driver_path` as the workaround)
2. `convert_and_write(raw_csv, dest)` — converts CelesTrak format to `.swbin` v2 (2 raw columns: `f10`, `kp`) using PCHIP interpolation

**Conversion stub** — the online conversion's HTTP fetch is stubbed until implemented. Use `driver_path` in `rope.conf`, or `--driver` on `rope forecast`, to point to a pre-converted `.swbin`/CSV as a workaround.

**Data resolution priority** (in `StackedEnsemblePipeline::load_sw_db()`):
```
1. Config.driver_path is set → use it directly (no cache, no network)
2. manifest.drivers.source is set → DriverCacheManager::get_path(source)
3. Neither → throw "no driver_path set and manifest has no 'drivers.source'"
```
`Config.driver_path` is reachable two ways: `paths.driver_path` in `rope.conf`, or the CLI's `--driver <path>` flag (which sets it directly, bypassing the config file). Either way this is **all-or-nothing** — a full replacement of the driver data source, not a merge; if the model needs a raw column the supplied file doesn't have, resolution fails loudly at the point that column is requested rather than falling back to any other source.

---

## Binary formats

### `.swbin` — space weather binary (v2, self-describing)

`include/rope/io/driver_bin.h`, `src/io/driver_bin.cpp`

```
Header (16 bytes, little-endian):
  uint32  magic    = 0x52505357  ("RPSW")
  uint32  version  = 2
  uint32  nrows
  uint32  ncols                        -- count of raw columns

Name table (ncols entries, immediately after the header):
  uint32  name_len
  char    name[name_len]               -- not NUL-terminated

Records (nrows × (8 + 4*ncols) bytes):
  int64   tp                            -- Unix timestamp
  float32 col[0..ncols)                  -- one value per name-table entry, in order
```

This serializes the same generic `raw_names_`/`raw_data_` representation `SpaceWeatherDB` already holds in memory — no fixed column set, and `t1`–`t4`/`doy`/`hour_int` are never stored (always derived or backfilled at load time, per `DriverRow::get()`). Clean break from v1 (pre-1.0; no dual-version support) — regenerate any existing `.swbin` via `rope convert-sw` from its source CSV.

~17 MB for 736 K rows (1957–present, 2 raw columns), versus ~39 MB CSV. Convert via:
```
rope convert-sw --input sw_celestrack_1957.csv --output sw_celestrack_1957.swbin
```

### `.icbin` — IC table binary (v2, self-describing)

`include/rope/io/ic_bin.h`, `src/io/ic_bin.cpp`

```
Header (16 bytes, little-endian):
  uint32  magic      = 0x52504943  ("RPIC")
  uint32  version    = 2
  uint32  nrows
  uint32  latent_dim  (K)

Axis name table (exactly 2 entries, immediately after the header):
  uint32  name_len
  char    name[name_len]               -- not NUL-terminated

Records (nrows × (2 + K) × 4 bytes):
  float32 axis0_value
  float32 axis1_value
  float32 y[K]
```

Axis identity is not fixed to F10/Kp — whatever 2 names the source table declares, cross-checked at pipeline-construction time against the manifest's `ic.params.grid_axes` (throws loudly on mismatch — the artifact and its own documented axes must agree). Convert via:
```
rope convert-ic --input IC_Table_modified.csv --output ic_table.icbin
```

The `from_file()` factory on both `SpaceWeatherDB` and `ICTable` dispatches on file extension (`.swbin` / `.icbin` → binary, anything else → CSV).

---

## IC table

`include/rope/io/ic_table.h`, `src/io/ic_table.cpp`

Maps a 2-axis grid (commonly F10, Kp, but declared per-model — see below) to K-dimensional latent initial conditions. Used to seed `mu_lat[0]` (the t=0 snapshot) without model inference.

**Axis names are auto-detected, not hardcoded:** the CSV constructor treats any header that isn't `y<N>` as an axis column — exactly 2 are required, in whichever order the file declares them (`csv.column_names()`, name-keyed, order-independent). `ICTable::axis_names()` exposes what it found.

**Interpolation strategy:**
1. Build unique sorted axes from the table's grid points
2. `get_latent_coeffs(axis_values)` → bilinear interpolation on the 2-axis grid if the query is within the convex hull (`axis_values.size()` must equal `axis_names().size()`, checked via `std::span` — 2 today; a higher-dimensional IC kind would need its own `ic.kind`, not a variable-length axis list here)
3. Falls back to nearest-neighbour if outside the hull

`grid_axes` is declared in `model_manifest.json`'s top-level `ic.params.grid_axes` (required, exactly 2 entries); `latent_dim` is the manifest's top-level `latent_dim`. Loaded behind the `IICSource` interface via `make_ic_source(dir, manifest.ic_kind)` (`src/forecast/backends/`) — `IICSource::axis_names()` and `get_latent_coeffs(span)` are likewise axis-name-agnostic. See `model-registry.md` for the full `ic` block schema.

The IC table is auto-discovered from `exported_dir`: tries `ic_table.icbin` first, then `ic_table.csv`.

**Known limitation:** regardless of what a model's `drivers.columns` declares, its raw driver source must still separately supply whatever names `ic.params.grid_axes` references — `StackedEnsemblePipeline` resolves each axis value via `DriverRow::get(axis_name)` at the same points it resolves driver features, so an axis name that isn't in the raw data (and isn't one of the derived names) fails loudly at that point, not at manifest-validation time.

---

## Preprocessing script

`scripts/preprocess_sw.py` converts a CelesTrak SW CSV (or the CSSI fixed-width `.txt` format) to `.swbin` using PCHIP interpolation:
- F10.7: daily → hourly via `scipy.interpolate.PchipInterpolator`
- Kp: 3-hourly → hourly per day (raw values in tenths ÷ 10), using next day's KP1 as the 24h endpoint for continuity

Supports `--input`/`--url`, `--output`, `--input-format {auto,csv,txt}`, and `--no-predicted` (exclude predicted sections when using `.txt`). Like `driver_cache.cpp`, this is specific to the one named CelesTrak source (always producing `f10`/`kp`) — not a generic conversion tool.
