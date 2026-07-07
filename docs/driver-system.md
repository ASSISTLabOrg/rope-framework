# Driver System

The driver system loads hourly space weather data (F10.7 solar flux and Kp geomagnetic index) and makes it available to the pipeline as a sequence of `DriverRow` feature vectors. It also manages the IC table, which maps (F10, Kp) to initial latent conditions.

---

## DriverRow

Defined in `include/rope/io/driver_db.h`. Represents one hour of derived driver features:

| Field | Meaning |
|-------|---------|
| `tp` | UTC timestamp (seconds since Unix epoch) |
| `f10` | F10.7 solar flux (SFU) |
| `kp` | Kp geomagnetic index |
| `t1` | `sin(2π·hour/24)` — diurnal harmonic |
| `t2` | `cos(2π·hour/24)` — diurnal harmonic |
| `t3` | `sin(2π·doy/365.25)` — annual harmonic |
| `t4` | `cos(2π·doy/365.25)` — annual harmonic |
| `doy` | Continuous day-of-year (`int_doy + hour/24`) |
| `hour_int` | Hour of day [0–23] |

`t1`–`t4` and `doy` are **not** stored in the binary file — they are recomputed from `tp` by `harmonics()` in `datetime.h` every time `make_row()` is called.

---

## SpaceWeatherDB

`include/rope/io/driver_db.h`, `src/io/driver_db.cpp`

An in-memory column-oriented store of hourly space weather data. Loads at server startup; never modified at runtime.

**Internal layout:**
```
times_  : vector<TimePoint>   // sorted; binary-search index
f10_    : vector<float>
kp_     : vector<float>
doy_    : vector<float>
hour_   : vector<int>
```

**Loading:**
- `SpaceWeatherDB(csv_path)` — parses a CSV with columns `datetime`, `f10`, `kp`, optional `doy` and `hour`. Sorts by timestamp. Fills `doy`/`hour` from the timestamp when absent.
- `SpaceWeatherBin::load(bin_path)` → `SpaceWeatherDB` — reads `.swbin` binary directly into the column vectors.
- `SpaceWeatherDB::from_file(path)` — dispatches on `.swbin` extension vs. CSV.

**Lookup:** `lookup(tp)` runs `std::lower_bound` on `times_` (O(log N)), then calls `make_row()` which assembles a `DriverRow` and recomputes harmonics.

**DriverWindowBuilder:** builds the `(S-1)+(H+1)` contiguous hourly rows needed by the pipeline. Throws if any hour is missing — no gap-filling.

---

## DriverConfig

`include/rope/io/driver_config.h`, `src/io/driver_config.cpp`

Loaded from `<exported_dir>/driver_config.json`. Declares which columns the model was trained on:

```json
{
  "version": 1,
  "columns": ["f10", "kp", "t1", "t2", "t3", "t4"],
  "source": "celestrak_sw"
}
```

`columns` replaces the earlier implicit `DriverCols::Six/Seven` inference from `stats_ts.bin`. Valid names are the fields of `DriverRow`: `f10`, `kp`, `t1`, `t2`, `t3`, `t4`, `doy`, `hour_int`.

`source` names a registered online source (see `DriverCacheManager`). If absent, only `driver_path` can supply data.

When `driver_config.json` is absent, `Pipeline::load()` falls back to inferring columns from `stats_ts.bin`'s `driver_dim` (6 → `[f10,kp,t1,t2,t3,t4]`, 7 → adds `doy`).

---

## DriverCacheManager

`include/rope/io/driver_cache.h`, `src/io/driver_cache.cpp`

Manages a local cache of `.swbin` files refreshed from online sources. Integrated into `Pipeline::load()` so auto-refresh fires for every entry point (CLI, C API, Python, C#).

**Source registry** (add entries in `driver_cache.cpp`):

| Key | URL |
|-----|-----|
| `celestrak_sw` | `https://celestrak.org/SpaceData/SW-Last5Years.csv` |
| `celestrak_sw_all` | `https://celestrak.org/SpaceData/SW-All.csv` |

**`get_path(source)`** returns the path to a fresh `.swbin` file:
1. If the cached file is absent or older than `max_age_hours` → calls `refresh(source, dest)`
2. If refresh fails but a stale file exists → logs a warning and returns the stale file
3. If no file at all → throws

**`refresh(source, dest)`:**
1. `download(url)` — HTTP GET (not yet implemented; throws with a clear message pointing to `driver_path` as the workaround)
2. `convert_and_write(raw_csv, dest)` — converts CelesTrak format to `.swbin` using PCHIP interpolation (not yet implemented; same stub)

**Conversion stub** — the online conversion is stubbed until the Ap→Kp formula and 3-hourly→hourly interpolation are implemented. Use `driver_path` in `rope.conf` to point to a pre-converted `.swbin` as a workaround.

**Data resolution priority** (in `Pipeline::load()`):
```
1. Config.driver_path is set → use it directly (no cache, no network)
2. driver_config.json has "source" → DriverCacheManager::get_path(source)
3. Neither → throw "no driver data source configured"
```

---

## Binary formats

### `.swbin` — space weather binary

`include/rope/io/driver_bin.h`, `src/io/driver_bin.cpp`

```
Header (16 bytes, little-endian):
  uint32  magic    = 0x52505357  ("RPSW")
  uint32  version  = 1
  uint32  nrows
  uint32  reserved = 0

Records (nrows × 24 bytes):
  int64   tp        // Unix timestamp
  float32 f10
  float32 kp
  float32 doy       // continuous = int_doy + hour/24
  int32   hour_int
```

~17 MB for 736 K rows (1957–present), versus ~39 MB CSV. Convert via:
```
rope convert-sw --input sw_celestrack_1957.csv --output sw_celestrack_1957.swbin
```

### `.icbin` — IC table binary

`include/rope/io/ic_bin.h`, `src/io/ic_bin.cpp`

```
Header (20 bytes):
  uint32  magic      = 0x52504943  ("RPIC")
  uint32  version    = 1
  uint32  nrows
  uint32  latent_dim  (K)
  uint32  reserved

Records (nrows × (2 + K) × 4 bytes):
  float32 f10
  float32 kp
  float32 y[K]
```

Convert via:
```
rope convert-ic --input IC_Table_modified.csv --output ic_table.icbin
```

The `from_file()` factory on both `SpaceWeatherDB` and `ICTable` dispatches on file extension (`.swbin` / `.icbin` → binary, anything else → CSV).

---

## IC table

`include/rope/io/ic_table.h`, `src/io/ic_table.cpp`

Maps (F10, Kp) → K-dimensional latent initial conditions. Used to seed `mu_lat[0]` (the t=0 snapshot) without model inference.

**Interpolation strategy:**
1. Build unique sorted axes for F10 and Kp from the table's grid points
2. `get_latent_coeffs(f10, kp)` → bilinear interpolation on the (F10, Kp) grid if the query is within the convex hull
3. Falls back to nearest-neighbour if outside the hull

**IcConfig** (`include/rope/io/ic_config.h`) declares `grid_axes` and `latent_dim` in `<exported_dir>/ic_config.json`. When absent, defaults to `["f10", "kp"]` and `K=10`. The table auto-detects K from the CSV's column count (`y1, y2, …`).

The IC table is auto-discovered from `exported_dir`: tries `ic_table.icbin` first, then `ic_table.csv`.

---

## Preprocessing script

`scripts/preprocess_sw.py` converts a CelesTrak SW CSV (or the CSSI fixed-width `.txt` format) to `.swbin` using PCHIP interpolation:
- F10.7: daily → hourly via `scipy.interpolate.PchipInterpolator`
- Kp: 3-hourly → hourly per day (raw values in tenths ÷ 10), using next day's KP1 as the 24h endpoint for continuity

Supports `--input`/`--url`, `--output`, `--input-format {auto,csv,txt}`, and `--no-predicted` (exclude predicted sections when using `.txt`).
