# Grid and Interpolation

The `GridInterpolator` turns the discrete forecast grid into a continuous function over (time, LST, latitude, altitude). It lives in `include/rope/interpolate/grid_interpolator.h` and `src/interpolate/grid_interpolator.cpp`.

---

## ForecastGrid

Defined in `include/rope/core/types.h`. The in-memory result of a pipeline run:

```cpp
struct ForecastGrid {
    std::vector<float>        density;      // H * GRID_VOXELS float32
    std::vector<float>        uncertainty;  // H * GRID_VOXELS float32
    std::vector<TimePoint>    times;        // H Unix timestamps (hourly)
    int H = 0;
};
```

**Grid dimensions:**

| Axis | Points | Range | Step |
|------|--------|-------|------|
| LST | 72 | [0, 24) hours | 20 min |
| Latitude | 36 | [−87.5, 87.5]° | 5° |
| Altitude | 45 | [100, 980] km | 20 km |

Total voxels per time step: `GRID_VOXELS = 72 × 36 × 45 = 116,640`.

**Memory layout:** row-major `[t, lst, lat, alt]` with altitude as the fastest axis. Flat index:
```
idx(t, lst, lat, alt) = t * 116640 + lst * (36*45) + lat * 45 + alt
```

**Times** cover `[start+1h, start+H]` — the H genuine predictions. The initial condition at `start+0h` is internal to the pipeline and not included in the grid.

---

## GridInterpolator

Constructed from a `ForecastGrid`. Holds a read-only reference to the grid's flat arrays. Construction is O(N): validates the grid (throws on empty or all-zero) and builds the LST axis.

**Thread safety:** `query_interp` and `query_hold` are const and safe to call concurrently.

---

## Coordinate handling

### LST (Local Solar Time)

LST is **periodic** with period 24 hours. Any query value is first normalized to `[0, 24)` using:
```cpp
lst = lst - std::floor(lst / 24.0) * 24.0;
```
No out-of-range error is raised for LST.

### Latitude

Hard bounds: [−87.5, 87.5]. Requests outside this range throw `SpatialOutOfRangeError`. The 5° grid resolution means the extreme cells are at ±85°, and queries in [±85°, ±87.5°] extrapolate slightly but the grid is designed to be physically valid there.

### Altitude

Hard bounds: [100, 980] km. Requests outside this range throw `SpatialOutOfRangeError`.

### Time

Hard bounds: `[times[0], times[H-1]]`. Requests outside this range throw `TimeOutOfRangeError`.

---

## Spatial interpolation (`spatial_interp`)

Performed **independently per time step** in **log₁₀ space**:

1. **LST axis:** uniform step (24/72 h). Compute the fractional cell index, extract the two bounding cells (`li0`, `li1`), compute weight `wl`. The last LST cell wraps to the first (`li1 = (li0+1) % 72`).

2. **Latitude axis:** uniform step (175/35 = 5°). Same approach; no wrapping.

3. **Altitude axis:** uniform step (880/44 = 20 km). Same approach; no wrapping.

4. **Trilinear interpolation** in log₁₀ space across the 8 surrounding voxels:
   ```
   log_val = Σ w[i] * log10(density[cell_i])   for i in the 8 corners
   physical = 10^log_val
   ```
   Interpolating in log₁₀ space preserves the exponential density profile across altitude (which spans several orders of magnitude) and keeps the result physically non-negative.

5. The same trilinear weights are applied to `uncertainty` without the log transform — uncertainty is interpolated linearly.

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

The server catches these separately and returns specific error codes to the client (`time_out_of_range`, `spatial_out_of_range`).

---

## Accuracy notes

- **Spatial:** Trilinear interpolation in log₁₀ space is accurate for density profiles that are approximately log-linear in altitude (a reasonable assumption for the upper atmosphere). For highly structured features (e.g. storm-time density enhancements), there can be smoothing artefacts near the grid boundaries.
- **Temporal:** Linear blending is adequate for the 1-hour grid cadence. Sub-minute queries within an hour are well-served.
- **LST periodicity:** The interpolation wraps the 72nd LST cell back to the 0th, which is correct for a zonally averaged model. The physical field is smooth at the wrap point.
