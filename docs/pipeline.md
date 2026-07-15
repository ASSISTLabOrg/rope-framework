# Forecast Pipeline

The pipeline is implemented as `StackedEnsemblePipeline` (selected by pipeline kind in `model_manifest.json`), a concrete subclass of the abstract `Pipeline` interface in `include/rope/forecast/pipeline.h`. Each `rope forecast` invocation constructs it fresh via `forecast::load(cfg)`.

`run_streaming(start_iso, horizon, chunk_hours, sink, latent_sink)` is the pure-virtual method each kind implements. `run(start_iso, horizon, latent_mean_out)` is concrete on the base class, collecting `run_streaming()`'s chunks into one `ForecastGrid`. `rope forecast` calls `run_streaming()` directly; tests/tooling wanting one in-memory grid call `run()`.

---

## Pipeline constants

| Constant | Typical value | Meaning |
|----------|---------------|---------|
| `K` | 10 | Latent space dimension (read from manifest) |
| `S` | 3 | Sequence length (read from manifest) |
| `M` | 15 | Base model count (read from manifest) |
| `D` | 16–17 | Total feature dim (K + driver_dim); inferred from `stats_ts.bin` |
| `DECODE_BATCH` | 120 | Max latent vectors per decoder call (read from manifest) |

---

## Config fields

Defined in `include/rope/forecast/pipeline.h`.

| Field | Default | Meaning |
|-------|---------|---------|
| `exported_dir` | — | Directory of model artifacts |
| `driver_path` | empty | Explicit driver file; bypasses cache manager |
| `cache_dir` | platform default | Where `DriverCacheManager` stores `.swbin` files |
| `cache_max_age_hours` | 24 | Max cached driver file age |
| `intra_threads_base` | 1 | ORT intra-op threads for base models |
| `intra_threads_meta` | 0 (→ hw_concurrency) | ORT intra-op threads for meta model |
| `intra_threads_decoder` | 0 (→ hw_concurrency) | ORT/LibTorch threads for decoder |
| `decoder_device` | `"cpu"` | LibTorch device string for decoder |
| `compute_uncertainty` | `true` | When false, skips UT; sets uncertainty to zero |
| `decode_chunk_hours` | `72` | Bounds peak memory during decode (see "Streaming decode" below). `<=0` = one chunk covering the whole horizon |
| `log` | nullptr | `std::function<void(std::string_view)>` called with load progress |

---

## Initialization

Called once per `rope forecast` invocation. Loads all artifacts from `exported_dir`.

```
Load model_manifest.json  → ModelManifest (paths, columns, latent_dim, …)
Load stats_ts.bin         → FeatureNormalizer (K, D, driver_dim)
Load ic_table.icbin or ic_table.csv → ICTable (bilinear + nearest-neighbour)
Resolve driver data       → SpaceWeatherDB::from_file(driver_path or cache)
Load M base models        → vector<IModel>
Load meta_model.onnx      → EnsembleFuser
Load decoder stage(s)     → vector<LatentDecoder>
Construct SlidingWindowRollout
```

Driver data resolution: explicit `driver_path` → `DriverCacheManager::get_path(source)` → error. See [driver-system.md](driver-system.md).

---

## `run_streaming(start_iso, H, chunk_hours, sink, latent_sink)` — step by step

### Step 1: Build the driver window

```cpp
all_rows = DriverWindowBuilder::build(*sw_db_, start_iso, H+1, S);
// Returns S+H rows chronologically.

hist_rows  = all_rows[0 .. S-1]    // S rows ending at start_iso
fcast_rows = all_rows[S-1 .. S+H]  // H+1 rows starting at start_iso
```

Each `DriverRow` carries `f10`, `kp`, `t1`–`t4` (harmonics), `doy`, `hour_int`. The extra forecast row is needed because the rollout slides one step past the last prediction.

### Step 2: Build the initial sequence

```
X_init (S, D) — normalized from hist_rows
```

For each of the S history timesteps:
1. IC table lookup at `(f10, kp)` → K-D latent coefficient vector
2. Fill driver features by column name (from manifest)
3. Z-score normalize the full `(K + driver_dim)` vector via `FeatureNormalizer`

### Step 3: Build the driver chunk

```
x_chunk (H+1, S, D) — normalized sliding windows
```

Each window is the previous shifted by one timestep, with:
- The **latent** part of the last row zeroed (overwritten by model output)
- The **driver** part filled from `fcast_rows[t]` and normalized

### Step 4: Base model rollout (parallel)

```
base_latents_norm (M, H, K) — H normalized predictions per model
```

Each base model runs `SlidingWindowRollout::run(model, x_chunk, H, out)` independently. OpenMP parallelizes across models.

Rollout loop (`t = 1..H`):
1. Feed `inp (S, D)` → model → `pred (K)`  [normalized latent]
2. Store `pred` at `out[(t-1)*K]`
3. If `t < H`: slide window — `inp[0:S-1] ← inp[1:S]`, `last_row.latent ← pred`, `last_row.driver ← x_chunk[t+1] drivers`

`try_infer_into()` on ONNX models binds I/O buffers once per rollout to avoid per-step allocation.

### Step 5: Meta model fusion

```cpp
FusionResult meta_out = meta_model_->fuse(x_chunk.data(), T, base_latents_norm.data());
// fuse() packs x_chunk + base_latents into the flat ONNX input internally.
meta_mean_norm (H, K) = meta_out.mean
```

### Step 6: Build `mu_lat`

```
ts_norm.denorm_latents_block(meta_mean_norm, H)   // back to physical space

init_lat (K) = X_init[S-1, 0:K] denormalized     // IC at t=0

mu_lat (H+1, K):
    mu_lat[0]    = init_lat          // initial condition
    mu_lat[1..H] = meta_mean_norm    // H model predictions
```

`latent_sink`, if passed, fires here once with `mu_lat[K..(H+1)*K)` (IC row dropped), before Step 7.

### Step 7: Decode — the only chunked step

Steps 1-6 always run whole-horizon (cheap, latent-space). Only decode is chunked, per `[t_lat, t_lat+count_lat)` of `mu_lat`'s `H+1` rows (`count_lat <= chunk_hours`):

```
# uncertainty off
density = decoder.decode(mu_lat[t_lat:t_lat+count_lat], count_lat, K)
uncertainty = zeros(count_lat × voxels)

# uncertainty on (sigma-point construction still runs once, whole-horizon)
sigma_slice = ut.sigma_lat[t_lat*N_SIG : (t_lat+count_lat)*N_SIG]
dens_sigmas = decoder.decode(sigma_slice, count_lat*N_SIG, K)
density_mean, uncertainty = weighted mean/variance over dens_sigmas via ut.Wm/ut.Wc
```

### Step 8: Deliver each chunk to `sink`

The first chunk (`t_lat==0`) drops its IC row before delivery, so it may emit one fewer row than `chunk_hours`. `t_offset` is H-indexed, matching `ForecastGrid::density`/`uncertainty`/`times`. Chunks arrive in increasing, contiguous order summing to `H`. `rope forecast` feeds each chunk straight to `ForecastGridBinWriter`/`ForecastZarrWriter`; the full grid is never held in memory.

---

## Streaming decode (`decode_chunk_hours`)

`chunk_hours <= 0` = one chunk (whole horizon). Otherwise:

```
peak_bytes ≈ chunk_hours × N_SIG × voxels × 4 × 2   (uncertainty on, N_SIG = 2K+1)
peak_bytes ≈ chunk_hours × voxels × 4                (uncertainty off)
```

Reference model (`K=10`, 72×36×45 grid): ~19.6 MB/chunk-hour with uncertainty on. Default `decode_chunk_hours=72` ≈ 1.4 GB peak. `rope forecast` logs the computed estimate at start.

This bounds the decode buffers only, not total process memory. Profiling found model loading is cheap (~0.1 GB) but the first inference call adds a largely fixed several-GB jump (observed ~2.7 GB at horizon=1) consistent with ONNX Runtime's/LibTorch's own arena allocator, not anything `decode_chunk_hours` controls.

---

## Component files

| Component | Files |
|-----------|-------|
| Public interface | `include/rope/forecast/pipeline.h`, `src/forecast/pipeline.cpp` |
| Kind dispatch | `src/forecast/pipeline_registry.h/.cpp` |
| Pipeline implementation | `src/forecast/stacked_ensemble/stacked_ensemble_pipeline.h/.cpp` |
| Rollout strategy | `src/forecast/stacked_ensemble/rollout_strategy.h`, `src/forecast/stacked_ensemble/sliding_window_rollout.h` |
| Ensemble fuser | `src/forecast/stacked_ensemble/ensemble_fuser.h` |
| Latent decoder | `src/forecast/stacked_ensemble/latent_decoder.h` |
| Unscented Transform | `src/forecast/stacked_ensemble/unscented_transform.h/.cpp` |
| Grid stitching | `src/forecast/stacked_ensemble/grid_stitch.h` |
| Model interface | `src/forecast/backends/model_interface.h` |
| Model factory | `src/forecast/backends/model_factory.cpp` |
| ONNX backend | `src/forecast/backends/onnx_model.h` |
| LibTorch backend | `src/forecast/backends/libtorch_model.h` |
| Runtime version check | `src/forecast/backends/runtime_compat.h/.cpp` |

`stacked_ensemble/` holds everything private to this one pipeline kind; `backends/` holds the ONNX/LibTorch `IModel` abstraction, shared by any kind that runs neural-net inference. See [adding-a-pipeline.md](adding-a-pipeline.md) for the convention a new kind follows.

---

## Performance notes

- **Base model rollout** is the dominant compute cost. OpenMP parallelizes across M models; always whole-horizon.
- **Decoder** has two nested chunking levels: `decode_chunk_hours` (outer, voxel-space memory) and `DECODE_BATCH` (inner, one ONNX/LibTorch call's size).
- **UT is optional.** `compute_uncertainty = false` skips it.
- **Driver lookup** is O(log N) binary search on sorted timestamps.
