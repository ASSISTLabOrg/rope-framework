# Version Control

## Branch model

| Branch | Purpose |
|--------|---------|
| `main` | Release snapshots only. Every commit on `main` corresponds to a tagged release. Never pushed to directly — the release workflow handles it. |
| `develop` | Primary development branch. All feature and bugfix branches merge here. CI builds all platform variants on every push. |
| `feature/<name>` | New work branched off `develop`. CI builds Linux CPU only. Merge back to `develop` via PR. |
| `bugfix/<platform>/<name>` | Platform-targeted fixes branched off `develop`. CI builds only the affected platform(s). Merge back to `develop` via PR. |

## Branch naming

`feature/` — anything that adds new behaviour, refactors, or extends the public API.

`bugfix/<platform>/` — the platform segment must be one of:

| Segment | Builds |
|---------|--------|
| `linux` | linux-cpu, linux-cuda12 |
| `windows` | windows-cpu, windows-cuda12 |
| `macos` | macos-arm64 |
| `all-platforms` | all five variants |

CI enforces this — a branch named `bugfix/bad-name/fix` fails the plan step with an error.

## Releases

Tag `main` with `vX.Y.Z`. The release workflow picks up the tag, builds all platform variants, packs the .NET NuGet, and publishes a GitHub release. Version is injected at build time via `ROPE_RELEASE_VERSION`.
