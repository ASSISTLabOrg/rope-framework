# Adding a Pipeline Kind

## How the registry works

`forecast::load(cfg)` reads `model_manifest.json` from `exported_dir`, checks `manifest.kind`, and dispatches to the matching constructor via `create_pipeline_for_kind()` in `src/forecast/pipeline_registry.cpp`. There is no plugin system — it is a plain string switch. Adding a new kind requires a code change in this file.

`known_kinds()` in `pipeline_registry.h` is tested against the `stable` entries in `rope-registry/kinds.json` (Category A test `test_kind_registry.cpp`). Both must be updated in sync.

---

## Checklist

### 1. `rope-registry` — add the kind

Add an entry to `kinds.json`:
```json
{ "kind": "my_new_kind", "schema": "schemas/kinds/my_new_kind.schema.json", "status": "stable" }
```

Optionally add a JSON Schema at the path above (not required by the runtime, but useful for tooling).

### 2. `include/rope/io/model_manifest.h` — add a spec struct

```cpp
struct MyNewKindSpec {
    // fields the manifest parser will populate
};
```

Add an optional field to `ModelManifest`:
```cpp
// Present iff kind == "my_new_kind".
std::optional<MyNewKindSpec> my_new_kind;
```

### 3. `src/io/model_manifest.cpp` — parse the spec

Add a branch to the manifest loader that populates `my_new_kind` when `manifest.kind == "my_new_kind"`.

### 4. `src/forecast/` — write the pipeline

Create a new subdirectory `src/forecast/my_new_kind/` and put `my_new_kind_pipeline.h`/`.cpp` there, along with any helper classes private to this kind (rollout strategy, fuser, decoder, uncertainty propagation, etc. — whatever is analogous to `stacked_ensemble/`'s `ensemble_fuser.h`, `latent_decoder.h`, `unscented_transform.h`). Only `src/forecast/` itself holds kind-agnostic plumbing (`pipeline.cpp`, `pipeline_registry.*`, `config_builder.cpp`); `src/forecast/backends/` holds the ONNX/LibTorch `IModel` abstraction shared by any kind that runs neural-net inference — reuse it rather than duplicating a model-backend wrapper inside your new kind's subdirectory. A kind with no learned models at all (e.g. a closed-form matrix/ODE rollout) may not need `backends/` at all.

The pipeline class must:
- Inherit `Pipeline` from `include/rope/forecast/pipeline.h`
- Implement `ForecastGrid run(const std::string& start_iso, int horizon) override`
- Take `(const Config& cfg, const io::ModelManifest& manifest)` in its constructor

### 5. `src/forecast/pipeline_registry.cpp` — register it

```cpp
#include "my_new_kind/my_new_kind_pipeline.h"

// In create_pipeline_for_kind():
if (manifest.kind == "my_new_kind")
    return std::make_unique<MyNewKindPipeline>(cfg, manifest);
```

### 6. `src/forecast/pipeline_registry.h` — update `known_kinds()`

```cpp
inline std::vector<std::string> known_kinds() {
    return {"stacked_ensemble", "my_new_kind"};
}
```

### 7. `CMakeLists.txt` — add source files

```cmake
add_library(rope_forecast STATIC
    ...
    src/forecast/my_new_kind/my_new_kind_pipeline.cpp
)
```

---

## Constraints

- The `Pipeline` interface has only one method: `run(start_iso, horizon)`. Initialization happens entirely in the constructor.
- `Config` is the runtime config (`rope.conf` values). `ModelManifest` is the per-artifact spec. The pipeline constructor receives both.
- `forecast-invariants.md` rules apply to any new pipeline: uncertainty is mandatory, fail loudly, deterministic output.
- The `kind` string is part of the public contract (it appears in `model_manifest.json` alongside the artifacts). Treat changes to an existing kind's string as a breaking change.
