# Building from Source

## Prerequisites

| Tool | Minimum version | Notes |
|------|----------------|-------|
| CMake | 4.2 | Install via `pip install "cmake>=4.2"` |
| C++ compiler | GCC 12 / Clang 14 / MSVC 2022 | C++20 required |
| Ninja (optional) | any | Faster than Make; use `-G Ninja` |

Runtime dependencies (ONNX Runtime, LibTorch) are downloaded automatically at configure time when `ROPE_DOWNLOAD_DEPS=ON`. No manual installation required.

---

## Quick start (Linux / macOS)

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DROPE_DOWNLOAD_DEPS=ON \
  -DROPE_HARDWARE=cpu \
  -DROPE_USE_LIBTORCH=ON \
  -DROPE_BUILD_TESTS=ON

cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

---

## CMake options

### Core options

| Option | Default | Description |
|--------|---------|-------------|
| `ROPE_DOWNLOAD_DEPS` | `OFF` | Download ORT and LibTorch automatically during configure |
| `ROPE_HARDWARE` | `cpu` | Hardware variant: `cpu`, `cuda12`, `cuda11`, `rocm6` |
| `ROPE_USE_LIBTORCH` | `ON` | Build the TorchScript decoder backend (requires LibTorch) |
| `ROPE_BUILD_TESTS` | `OFF` | Build and register CTest unit/integration tests |
| `ROPE_BUILD_LEGACY` | `OFF` | Also build the legacy `rope_demo` target |

### Dependency paths (when `ROPE_DOWNLOAD_DEPS=OFF`)

| Option | Description |
|--------|-------------|
| `ONNXRUNTIME_ROOT` | Root directory of a manually installed ORT |
| `ORT_INC` | ORT include directory (overrides `ONNXRUNTIME_ROOT`) |
| `ORT_LIB` | ORT shared library path (overrides `ONNXRUNTIME_ROOT`) |
| `LIBTORCH_ROOT` | Root of a LibTorch distribution (sets `Torch_DIR`) |
| `Torch_DIR` | CMake config directory for LibTorch |

### Release / packaging

| Option | Description |
|--------|-------------|
| `ROPE_RELEASE_VERSION` | Semver string injected at build time (e.g. `1.2.3`). Overrides the hardcoded version in `CMakeLists.txt`. Used by CI on tagged releases. |
| `ROPE_PACKAGE_SUFFIX` | Suffix appended to the CPack archive name (e.g. `linux-x86_64-cpu`). |
| `ROPE_MODELS_DIR` | Alternative `exported/` directory to bundle in the package. |

---

## Hardware variants

### CPU (default)

```bash
cmake -B build -DROPE_DOWNLOAD_DEPS=ON -DROPE_HARDWARE=cpu -DROPE_USE_LIBTORCH=ON
```

Downloads the CPU-only builds of ORT and LibTorch. LibTorch is required for the TorchScript decoder backend; if you only need the ONNX decoder, set `ROPE_USE_LIBTORCH=OFF`.

### CUDA 12

```bash
cmake -B build -DROPE_DOWNLOAD_DEPS=ON -DROPE_HARDWARE=cuda12 -DROPE_USE_LIBTORCH=OFF
```

Downloads the CUDA 12 build of ORT. LibTorch CUDA is not auto-downloaded; supply `Torch_DIR` manually if you need GPU decoder support. The decoder device is set at runtime via `decoder.device = cuda` in `rope.conf`.

### ROCm 6 (Linux only)

ORT ROCm builds are not on GitHub Releases. Use `scripts/get-ort-libs.sh --hardware rocm6` to obtain them, then set `ONNXRUNTIME_ROOT` manually.

---

## Running tests

Tests are split into three categories:

**Category A — pure logic (no ORT required):**
```bash
ctest --test-dir build -R rope_tests --output-on-failure
```
Covers interpolation, CSV reader, config reader, datetime utilities, and version.

**Category B — pipeline integration (requires ORT):**
```bash
ctest --test-dir build -R rope_pipeline --output-on-failure
```
Loads stub ONNX models from `tests/fixtures/test_models/`, runs a full forecast, checks grid shape and physical validity. When LibTorch is available, the `.pt` fixture is generated at test time by `gen_decoder_pt`.

**Category C — CLI integration (requires ORT + Python + pytest):**
```bash
ctest --test-dir build -R rope_cli --output-on-failure
# or run directly:
ROPE_EXE=build/rope ROPE_FIXTURE_DIR=tests/fixtures pytest tests/python/ -v
```
Exercises the full CLI round-trip: server spawn, forecast, single-point query, batch query, idle timeout.

---

## Creating a release package

```bash
cmake --build build --parallel
cpack --config build/CPackConfig.cmake
```

Produces `rope_framework-<version>-<suffix>.tar.gz` (Linux/macOS) or `.zip` (Windows). The archive contains `bin/`, `lib/`, `include/`, `config/`, `dotnet/`, and `python/`. Model artifacts and driver data are **not** included — they are distributed separately.

---

## Build system layout

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Root CMake; defines all targets, install rules, CPack config |
| `cmake/Dependencies.cmake` | ORT and LibTorch download and path resolution |
| `cmake/version.h.in` | Template for the generated version header |
| `tests/cpp/CMakeLists.txt` | C++ test targets and CTest registration |

The targets defined in `CMakeLists.txt`:

| Target | Type | Purpose |
|--------|------|---------|
| `rope_core` | static lib | Platform layer, datetime, types |
| `rope_io` | static lib | CSV/binary I/O, config, stats, driver DB, IC table, cache manager |
| `rope_interpolate` | static lib | Grid interpolator |
| `rope_client` | static lib | IPC client (used by CLI and C API) |
| `rope_forecast` | static lib | Full inference pipeline (requires ORT) |
| `rope_server` | static lib | Server request router and cache |
| `rope` | shared lib | C API (`librope.so` / `.dylib` / `.dll`) |
| `rope_exe` | executable | CLI binary (`rope`) |
