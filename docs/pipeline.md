# Forecast Pipeline

The pipeline is implemented as `EnsembleFusionDecoderPipeline` (selected by pipeline kind in `model_manifest.json`), a concrete subclass of the abstract `Pipeline` interface in `include/rope/forecast/pipeline.h`. Constructed once at server startup via `forecast::load(cfg)`, then called repeatedly per request via `pipe->run(start_iso, horizon)`.

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
| `log` | nullptr | `std::function<void(std::string_view)>` called with load progress |

---

## Initialization

Called once at server startup. Loads all artifacts from `exported_dir`.

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

## `run(start_iso, H)` — step by step

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

### Step 7: Decode

**Fast path (uncertainty disabled):**
```
all_density = decoder.decode(mu_lat, H+1, K)  // (H+1) × GRID_VOXELS
density_mean = all_density[1..H]               // drop IC snapshot
uncertainty  = zeros(H × GRID_VOXELS)
```

**Full path (Unscented Transform):**

Propagates ensemble spread through the non-linear decoder. Hyperparameters: α=1, β=2, κ=0 → λ=0, c=K, N_SIG=2K+1=21. Implemented in `src/forecast/unscented_transform.h/.cpp`.

For each timestep `t`:
1. Compute sample covariance `Pt (K, K)` from M base-model predictions (Bessel-corrected)
2. Factorise `c·Pt` via Cholesky → lower triangle `L`
3. Construct N_SIG sigma points: `μ`, `μ ± L[:,i]` for `i=0..K-1`

All `(H+1) × N_SIG` sigma points decoded in one batched call. UT mean and variance accumulated with weights `Wm`, `Wc`.

```
density_mean (H, GRID_VOXELS)  = weighted mean  [drop t=0]
uncertainty  (H, GRID_VOXELS)  = sqrt(weighted variance)  [drop t=0]
```

### Step 8: Return ForecastGrid

```cpp
grid.H          = H;
grid.density    = std::move(density_mean);    // H × 116,640 float32
grid.uncertainty= std::move(uncertainty);
grid.times      = {fcast_rows[1].tp, ..., fcast_rows[H].tp};
```

Times span `[start+1h, start+H]`.

---

## Component files

| Component | Files |
|-----------|-------|
| Public interface | `include/rope/forecast/pipeline.h`, `src/forecast/pipeline.cpp` |
| Pipeline implementation | `src/forecast/ensemble_fusion_decoder_pipeline.h/.cpp` |
| Rollout strategy | `src/forecast/rollout_strategy.h`, `src/forecast/sliding_window_rollout.h` |
| Ensemble fuser | `src/forecast/ensemble_fuser.h` |
| Latent decoder | `src/forecast/latent_decoder.h` |
| Unscented Transform | `src/forecast/unscented_transform.h/.cpp` |
| Model interface | `src/forecast/model_interface.h` |
| Model factory | `src/forecast/model_factory.cpp` |
| ONNX backend | `src/forecast/onnx_model.h` |
| LibTorch backend | `src/forecast/libtorch_model.h` |

---

## Performance notes

- **Base model rollout** is the dominant cost. OpenMP parallelizes across M models; each rollout is sequential (auto-regressive).
- **Decoder** is batched. All `H` (or `(H+1)×N_SIG` for UT) latent vectors are decoded in one call, chunked by `DECODE_BATCH`.
- **UT is optional.** `compute_uncertainty = false` in `rope.conf` skips it. Decoder runs once instead of N_SIG times.
- **Driver lookup** is O(log N) binary search on sorted timestamps.
