# GitHub Actions

Workflows live in `.github/workflows/`. Reusable platform jobs (`_build-linux.yml`, `_build-macos.yml`, `_build-windows.yml`) are called by both public workflows.

---

## `build.yml` — CI on every non-main push

Triggered on push to any branch except `main`, or manually with `test_all_platforms` flag.

A **plan** job runs first and decides which platform matrix to build based on branch name:

| Branch | Platforms built |
|--------|-----------------|
| `develop` | all five |
| `feature/*` | linux-cpu only |
| `bugfix/linux/*` | linux-cpu, linux-cuda12 |
| `bugfix/windows/*` | windows-cpu, windows-cuda12 |
| `bugfix/macos/*` | macos-arm64 |
| `bugfix/all-platforms/*` | all five |

Invalid `bugfix/` names fail immediately in the plan step.

On `develop` pushes: also runs a **Pack .NET (NuGet, dev)** job after the three CPU platform builds, producing a `0.0.0-dev.<run_number>` NuGet artifact (retention: 1 day).

Dependency caches are saved to the Actions cache only on `develop` pushes.

---

## `release.yml` — publish on version tag

Triggered by pushing a `vX.Y.Z` tag (or manually).

1. Builds all five platform variants in parallel, each injecting `ROPE_RELEASE_VERSION` from the tag.
2. **Pack .NET** job assembles the native binaries from the three CPU artifacts and runs `dotnet pack` with the release version.
3. **Publish** job downloads all artifacts and creates a GitHub Release (with auto-generated release notes), then deletes the staging artifacts from the run.

CUDA variants are built and included in the release but excluded from the .NET NuGet (CPU-only for managed distribution).

---

## `cleanup.yml` — weekly cache purge

Runs every Saturday at 04:00 UTC (and on manual dispatch). Deletes Actions caches for branches that no longer exist, keeping caches for live branches and tags.
