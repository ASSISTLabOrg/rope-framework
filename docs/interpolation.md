# Grid and Interpolation

The `GridInterpolator` turns the discrete forecast grid into a continuous function over (time, LST, latitude, altitude). It lives in `include/rope/interpolate/grid_interpolator.h` and `src/interpolate/grid_interpolator.cpp`.

---

## ForecastGrid

Defined in `include/rope/core/types.h`. The in-memory result of a pipeline run:

```cpp
struct ForecastGrid {
    GridSpec                   shape;        // this model's grid shape (see below)
    std::vector<float>        density;      // H * shape.voxels() float32
    std::vector<float>        uncertainty;  // H * shape.voxels() float32
    std::vector<TimePoint>    times;        // H Unix timestamps (hourly)
    int H = 0;
};
```

**Grid dimensions are per-model**, declared in `model_manifest.json`'s top-level `grid` object and carried on `GridSpec` (`include/rope/core/types.h`) — different trained models may target different physical grids. LST's range is always the full 24h cycle by definition of what LST is; lat/alt each declare their own count and physical range. A typical/example shape:

| Axis | Points | Range | Step |
|------|--------|-------|------|
| LST | 72 | [0, 24) hours | 20 min |
| Latitude | 36 | [−87.5, 87.5]° | 5° |
| Altitude | 45 | [100, 980] km | 20 km |

giving `shape.voxels() = 72 × 36 × 45 = 116,640` for that particular model — not a fixed constant.

**Memory layout:** row-major `[t, lst, lat, alt]` with altitude as the fastest axis. Flat index:
```
idx(t, lst, lat, alt) = t * shape.voxels() + lst * (shape.n_lat*shape.n_alt) + lat * shape.n_alt + alt
```

**Times** cover `[start+0h, start+horizon]` inclusive — `H = horizon+1` rows, including the initial condition at hour 0.

---

## GridInterpolator

Constructed from a `ForecastGrid` and an optional `ExtrapolationOptions` (defaults to extrapolation on, `n_etp_pts=8`). Holds a read-only reference to the grid's flat arrays. Construction is O(N): validates the grid (throws on empty or all-zero), builds the LST axis, and precomputes the polar-cap and altitude-extrapolation fit arrays — see "Polar caps" and "Altitude extrapolation" below.

**Thread safety:** `query_interp` and `query_hold` are const and safe to call concurrently. `set_extrapolation_options` (and the C API's `rope_set_extrapolation`) is **not** safe to call concurrently with queries on the same instance/handle — same contract as `rope_open`/`rope_close`.

---

## Coordinate handling

### LST (Local Solar Time)

LST is **periodic** with period 24 hours. Any query value is first normalized to `[0, 24)` using:
```cpp
lst = lst - std::floor(lst / 24.0) * 24.0;
```
No out-of-range error is raised for LST.

### Latitude

Hard bounds are the true physical range `[-90, 90]`, not the per-model grid range. Within `[shape.lat_min_deg, shape.lat_max_deg]`, latitude interpolates against real grid rows as usual. Beyond that (out to ±90) it blends toward a precomputed polar-cap value instead — see "Polar caps" below. Only requests outside `[-90, 90]` throw `SpatialOutOfRangeError`.

### Altitude

`shape.alt_min_km` is always a hard floor — requests below it throw `SpatialOutOfRangeError` regardless of extrapolation settings. Above `shape.alt_max_km`, behavior depends on `ExtrapolationOptions`: with `extrapolate_altitude` on (the default), the interpolator log-linearly extrapolates out to a hard ceiling of 2000 km — see "Altitude extrapolation" below. With it off, or past 2000 km, requests throw `SpatialOutOfRangeError` just like latitude/time out-of-range requests.

### Time

Hard bounds: `[times[0], times[H-1]]`. Requests outside this range throw `TimeOutOfRangeError`.

---

## Spatial interpolation (`spatial_interp`)

Performed **independently per time step** in **log₁₀ space**:

1. **LST axis:** uniform step (`24 / shape.n_lst` hours). Compute the fractional cell index, extract the two bounding cells (`li0`, `li1`), compute weight `wl`. The last LST cell wraps to the first (`li1 = (li0+1) % shape.n_lst`).

2. **Latitude axis:** uniform step (`(lat_max_deg - lat_min_deg) / (n_lat-1)`) within the grid's own range; no wrapping. Beyond `lat_min_deg`/`lat_max_deg`, the "far" bracket point is a precomputed polar-cap value instead of a real row — see "Polar caps" below.

3. **Altitude axis:** uniform step (`(alt_max_km - alt_min_km) / (n_alt-1)`). Same approach; no wrapping.

4. **Trilinear interpolation** in log₁₀ space across the 8 surrounding voxels:
   ```
   log_val = Σ w[i] * log10(density[cell_i])   for i in the 8 corners
   physical = 10^log_val
   ```
   Interpolating in log₁₀ space preserves the exponential density profile across altitude (which spans several orders of magnitude) and keeps the result physically non-negative.

5. The same trilinear weights are applied to `uncertainty` without the log transform — uncertainty is interpolated linearly.

---

## Polar caps

Every real point at latitude 90 (or -90) is the same physical point, so its value can't legitimately depend on LST — but a model's grid stores one independent sample per LST cell at its boundary row regardless. `GridInterpolator`'s constructor collapses each boundary row to a single value per `(t, alt)` — the mean of `log10(density)` across all LST cells at that row (the standard technique for defining a scalar field at this kind of coordinate singularity: the zonal mean of the nearest ring, the same idea used for pole handling in numerical weather models). This runs once when the interpolator is built, not per query.

A query with `lat` beyond `[lat_min_deg, lat_max_deg]` then blends linearly (in log₁₀ space) between the real boundary row and this precomputed cap value, reaching the cap exactly at ±90 — where the result is, by construction, identical for every LST. Uncertainty gets the same treatment, from its own boundary row.

This only changes behavior for latitudes a model's grid doesn't natively cover (e.g. 87.5°–90° for a model trained on `[-87.5, 87.5]`) — queries inside the grid's own range are completely unaffected, and the cap values add negligible one-time memory (`H × n_alt` floats per pole, per field).

---

## Altitude extrapolation

Density vs. altitude is not log-linear across a model's whole trained range (real curvature — scale height grows with altitude), but it *is* well-approximated by a local log-linear fit near the top of the grid. `GridInterpolator`'s constructor fits an ordinary-least-squares line, in log₁₀ space, per `(t, lst, lat)` column, using the `n_etp_pts` grid altitude bins nearest `alt_max_km`. This runs once at construction (and again whenever `set_extrapolation_options`/`rope_set_extrapolation` changes `n_etp_pts`), not per query.

A query with `alt_km > alt_max_km` (and `extrapolate_altitude` on) evaluates that column's fitted line at `alt_km` instead of reading a real grid value, then proceeds through the same LST/latitude trilinear blend as an in-range query — so it composes correctly with polar-cap latitudes too: the polar caps carry their own separate fit (over the already-collapsed cap values) rather than reusing the real-column fit. Uncertainty is extrapolated the same way, from its own fit over the same window.

Controlled by `ExtrapolationOptions`:

| Field | Default | Notes |
|---|---|---|
| `extrapolate_altitude` | `true` | Toggle; off reverts to throwing past `alt_max_km` |
| `n_etp_pts` | `8` | Altitude bins used for the local fit; must be in `[2, shape.n_alt]` |

`GridInterpolator::kMaxExtrapolationAltKm = 2000.0` is a hardcoded ceiling, not part of `ExtrapolationOptions` — always enforced regardless of the toggle. Settable via the C++ constructor/`set_extrapolation_options`, the `rope_set_extrapolation` C API function, `rope.conf`'s `[interpolation]` section, or `rope get --no-extrapolate-altitude`/`--n-etp-pts`.

---

## Temporal interpolation (`query_interp`)

After computing `r0 = spatial_interp(t_before)` and `r1 = spatial_interp(t_after)`:

```
alpha = (t_query - t_before) / (t_after - t_before)   // in [0, 1]
density     = lerp(r0.density,     r1.density,     alpha)
uncertainty = lerp(r0.uncertainty, r1.uncertainty, alpha)
```

Temporal interpolation is **linear in physical density** (not log₁₀). Each time step is already in physical space after the spatial trilinear step, so linear blending is appropriate for the hourly cadence.

**`query_hold`** snaps to the nearest time step `≤ t_query` (floor), bypassing temporal blending entirely.

---

## Error types

```cpp
// Both defined in include/rope/interpolate/grid_interpolator.h
class TimeOutOfRangeError    : public std::runtime_error { ... };
class SpatialOutOfRangeError : public std::runtime_error { ... };
```

Both the CLI and the C API (`src/capi/rope.cpp`'s `classify_exception`) catch these separately and map them to distinct error codes (`ROPE_ERR_TIME_RANGE`, `ROPE_ERR_SPATIAL_RANGE`). `set_extrapolation_options` additionally throws `std::invalid_argument` for an out-of-range `n_etp_pts`, mapped by `rope_set_extrapolation` to `ROPE_ERR_BAD_ARG`.

---

## Accuracy notes

- **Spatial:** Trilinear interpolation in log₁₀ space is accurate for density profiles that are approximately log-linear in altitude (a reasonable assumption for the upper atmosphere). For highly structured features (e.g. storm-time density enhancements), there can be smoothing artefacts near the grid boundaries.
- **Altitude extrapolation:** the local fit (last `n_etp_pts` bins) tracks the boundary region well, but accuracy degrades with distance past `alt_max_km` — it is not a substitute for a model trained to a higher ceiling. Treat extrapolated values past a few hundred km beyond `alt_max_km` with increasing caution.
- **Temporal:** Linear blending is adequate for the 1-hour grid cadence. Sub-minute queries within an hour are well-served.
- **LST periodicity:** The interpolation wraps the 72nd LST cell back to the 0th, which is correct for a zonally averaged model. The physical field is smooth at the wrap point.
