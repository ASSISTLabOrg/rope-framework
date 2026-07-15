"""
Category C — end-to-end CLI integration tests.

Exercises the full rope pipeline: forecast (writes the cache file), single-
point query, batch query, and error paths -- all via the rope CLI binary.
No server process, no sockets: `rope forecast` runs inference directly and
writes a cache file; `rope get` reads it (memory-mapped).

Required env vars (injected by CTest; fall back to sensible defaults for
manual runs from the build directory):
  ROPE_EXE          path to the rope binary
  ROPE_FIXTURE_DIR  path to the tests/fixtures directory
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Derive sensible defaults so `pytest tests/python/` works from the project
# root without needing env vars set (CTest sets them explicitly in CI).
_project_root = Path(__file__).parent.parent.parent
_default_exe  = _project_root / "build" / (
    "rope.exe" if sys.platform == "win32" else "rope"
)

ROPE_EXE    = os.environ.get("ROPE_EXE", str(_default_exe))
FIXTURE_DIR = Path(os.environ.get("ROPE_FIXTURE_DIR",
                                   _project_root / "tests" / "fixtures"))

# Must match the sw_test.swbin coverage window used by the pipeline tests.
# sw_test.csv covers 2023-12-31T22:00:00 through 2024-01-01T03:00:00 (6 rows);
# horizon=3 with seq_len=3 requires (seq_len-1)+(horizon+1) = 6 rows.
FORECAST_START   = "2024-01-01 00:00:00"
FORECAST_HORIZON = 3
QUERY_TIME       = "2024-01-01T01:00:00"


def _rope(*args, timeout=60):
    return subprocess.run(
        [ROPE_EXE, *args],
        capture_output=True, text=True, timeout=timeout,
    )


def _write_conf(path: Path, exported_dir=None) -> None:
    # driver_path: explicit binary fixture — bypasses cache manager entirely.
    # ic_csv is no longer set: the IC table is auto-discovered from exported_dir
    # (test_models/ic_table.icbin, generated alongside the other fixtures).
    exported_dir = exported_dir or (FIXTURE_DIR / "test_models")
    path.write_text(
        f"[paths]\n"
        f"exported_dir = {exported_dir}\n"
        f"driver_path  = {FIXTURE_DIR / 'test_models' / 'sw_test.swbin'}\n"
    )


# ---------------------------------------------------------------------------
# Module-scoped forecast-cache fixture — one cache file shared across most
# tests. No teardown needed: there is no process to stop, just a file.
# ---------------------------------------------------------------------------

class _ForecastCache:
    def __init__(self, cache_path: str, conf: str):
        self.cache_path = cache_path
        self.conf = conf

    def run(self, *args, timeout=60):
        return _rope("--cache-path", self.cache_path, *args, timeout=timeout)


@pytest.fixture(scope="module")
def forecast_cache(tmp_path_factory):
    tmp   = tmp_path_factory.mktemp("rope_cli")
    cache = str(tmp / "forecast_grid.bin")
    conf  = tmp / "rope.conf"
    _write_conf(conf)

    result = _rope(
        "--cache-path", cache, "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", str(conf),
    )
    assert result.returncode == 0, f"forecast failed:\n{result.stderr}"

    yield _ForecastCache(cache, str(conf))


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------

def test_forecast_returns_ok(forecast_cache):
    result = forecast_cache.run(
        "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", forecast_cache.conf,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["status"] == "ok"
    assert "window_start" in data
    assert "window_end"   in data


def test_rope_binding_forecast_uses_custom_cache_path(tmp_path):
    """Regression test: Rope.forecast() must pass --cache-path when a custom
    cache_path is configured -- otherwise it silently falls back to the
    default per-user cache file, which could read/write state unrelated to
    this Rope instance.
    """
    sys.path.insert(0, str(_project_root / "python"))
    from rope import Rope

    lib_path = _project_root / "build" / "librope.so"
    assert lib_path.is_file(), "librope.so not built"

    conf = tmp_path / "rope_binding.conf"
    _write_conf(conf)
    custom_cache = str(tmp_path / "custom_forecast_grid.bin")

    r = Rope(lib_path=lib_path, exe_path=Path(ROPE_EXE), cache_path=custom_cache, config_path=conf)
    result = r.forecast(FORECAST_START, FORECAST_HORIZON)
    assert result["status"] == "ok"

    # The cache file must actually have been written at the custom path (not
    # the default per-user one) -- reading it directly via the CLI proves it.
    probe = _rope(
        "--cache-path", custom_cache, "get", "--mode", "interp",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert probe.returncode == 0


# ---------------------------------------------------------------------------
# get — single point
# ---------------------------------------------------------------------------

def test_get_interp_returns_valid_point(forecast_cache):
    result = forecast_cache.run(
        "get", "--mode", "interp",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["density"]     > 0
    assert data["uncertainty"] >= 0


def test_get_hold_returns_valid_point(forecast_cache):
    result = forecast_cache.run(
        "get", "--mode", "hold",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["density"]     > 0
    assert data["uncertainty"] >= 0


def test_get_time_out_of_range_fails(forecast_cache):
    result = forecast_cache.run(
        "get", "--mode", "interp",
        "--time", "2030-01-01T00:00:00",
        "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
        timeout=10,
    )
    assert result.returncode != 0


def test_get_without_forecast_fails_clearly(tmp_path):
    missing_cache = str(tmp_path / "never_written.bin")
    result = _rope(
        "--cache-path", missing_cache, "get", "--mode", "interp",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
        timeout=10,
    )
    assert result.returncode != 0
    assert "forecast" in result.stderr.lower()


# ---------------------------------------------------------------------------
# get — batch
# ---------------------------------------------------------------------------

def test_batch_get_from_csv(forecast_cache, tmp_path):
    csv = tmp_path / "queries.csv"
    csv.write_text(
        "YYYY,MM,DD,HH,MIN,SS,lst,lat,alt_km\n"
        "2024,01,01,01,00,00,12.0,45.0,400.0\n"
        "2024,01,01,01,30,00,6.0,-30.0,300.0\n"
    )
    out = tmp_path / "results.json"
    result = forecast_cache.run(
        "get", "--mode", "interp",
        "--file", str(csv), "--output", str(out),
        timeout=15,
    )
    assert result.returncode == 0
    rows = json.loads(out.read_text())
    assert len(rows) == 2
    for row in rows:
        assert row["density"]     > 0
        assert row["uncertainty"] >= 0


# ---------------------------------------------------------------------------
# Discard-on-reforecast invariant
# ---------------------------------------------------------------------------

def test_second_forecast_discards_first(tmp_path):
    """A new `rope forecast` fully replaces the cache file -- there is no
    server holding old state around, so a query valid only under the first
    forecast's window must fail once a second, differently-shaped forecast
    has been written to the same cache path.
    """
    cache = str(tmp_path / "forecast_grid.bin")
    conf  = tmp_path / "rope.conf"
    _write_conf(conf)

    # horizon=3 -> window covers 2024-01-01T01:00:00 .. 03:00:00
    first = _rope(
        "--cache-path", cache, "forecast",
        "--start", FORECAST_START, "--horizon", "3", "--config", str(conf),
    )
    assert first.returncode == 0

    late_time = "2024-01-01T03:00:00"
    probe_a = _rope(
        "--cache-path", cache, "get", "--mode", "interp",
        "--time", late_time, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert probe_a.returncode == 0, "sanity check: first forecast's window must include late_time"

    # horizon=1 -> window covers only 2024-01-01T01:00:00
    second = _rope(
        "--cache-path", cache, "forecast",
        "--start", FORECAST_START, "--horizon", "1", "--config", str(conf),
    )
    assert second.returncode == 0

    probe_b = _rope(
        "--cache-path", cache, "get", "--mode", "interp",
        "--time", late_time, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
        timeout=10,
    )
    assert probe_b.returncode != 0, "old (horizon=3) window must be gone after re-forecasting with horizon=1"


# ---------------------------------------------------------------------------
# Pipeline load failure — a failed forecast never touches an existing cache
# ---------------------------------------------------------------------------

def test_bad_exported_dir_forecast_fails_but_existing_cache_survives(tmp_path):
    cache    = str(tmp_path / "forecast_grid.bin")
    good_conf = tmp_path / "rope_good.conf"
    _write_conf(good_conf)

    first = _rope(
        "--cache-path", cache, "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", str(good_conf),
    )
    assert first.returncode == 0, f"initial forecast failed:\n{first.stderr}"

    # Now point at a directory with no model artifacts at all -- the pipeline
    # fails to load. That failure must not touch the cache file written above.
    bad_conf  = tmp_path / "rope_bad.conf"
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    _write_conf(bad_conf, exported_dir=empty_dir)

    second = _rope(
        "--cache-path", cache, "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", str(bad_conf),
    )
    assert second.returncode != 0

    # The cache file from the first, successful forecast must still be intact.
    probe = _rope(
        "--cache-path", cache, "get", "--mode", "interp",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert probe.returncode == 0, "a failed forecast must not clobber the existing cache file"
