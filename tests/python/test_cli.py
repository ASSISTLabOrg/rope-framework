"""
Category C — end-to-end CLI integration tests.

Exercises the full rope pipeline: server spawn, forecast, single-point query,
batch query, idle timeout, and clean shutdown — all via the rope CLI binary.

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


def _write_conf(path: Path, idle_timeout_seconds: int = 30) -> None:
    # driver_path: explicit binary fixture — bypasses cache manager entirely.
    # ic_csv is no longer set: the IC table is auto-discovered from exported_dir
    # (test_models/ic_table.icbin, generated alongside the other fixtures).
    path.write_text(
        f"[paths]\n"
        f"exported_dir = {FIXTURE_DIR / 'test_models'}\n"
        f"driver_path  = {FIXTURE_DIR / 'test_models' / 'sw_test.swbin'}\n"
        f"[server]\n"
        f"idle_timeout_seconds = {idle_timeout_seconds}\n"
    )


# ---------------------------------------------------------------------------
# Module-scoped server fixture — one server shared across most tests.
# ---------------------------------------------------------------------------

class _Server:
    def __init__(self, sock: str, conf: str):
        self.sock = sock
        self.conf = conf

    def run(self, *args, timeout=60):
        return _rope("--socket", self.sock, *args, timeout=timeout)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp  = tmp_path_factory.mktemp("rope_cli")
    sock = str(tmp / "rope.sock")
    conf = tmp / "rope.conf"
    _write_conf(conf)

    result = _rope(
        "--socket", sock, "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", str(conf),
    )
    assert result.returncode == 0, f"server startup failed:\n{result.stderr}"

    srv = _Server(sock, str(conf))
    yield srv

    _rope("--socket", sock, "exit", timeout=10)


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------

def test_forecast_returns_ok(server):
    result = server.run(
        "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", server.conf,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["status"] == "ok"
    assert "window_start" in data
    assert "window_end"   in data


def test_rope_binding_forecast_uses_custom_socket_path(tmp_path):
    """Regression test: Rope.forecast() must pass --socket when a custom
    socket_path is configured -- otherwise it silently falls back to the
    default per-user socket, which can talk to a stale/unrelated server
    (see python/rope.py's forecast(), which previously omitted --socket
    unlike .open()/.shutdown()).
    """
    sys.path.insert(0, str(_project_root / "python"))
    from rope import Rope

    lib_path = _project_root / "build" / "librope.so"
    assert lib_path.is_file(), "librope.so not built"

    conf = tmp_path / "rope_binding.conf"
    _write_conf(conf)
    custom_socket = str(tmp_path / "custom.sock")

    r = Rope(lib_path=lib_path, exe_path=Path(ROPE_EXE), socket_path=custom_socket, config_path=conf)
    result = r.forecast(FORECAST_START, FORECAST_HORIZON)
    assert result["status"] == "ok"

    # The server must actually be listening on the custom socket (not the
    # default per-user one) -- talking to it directly via the CLI proves it.
    probe = _rope(
        "--socket", custom_socket, "get", "--mode", "interp",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert probe.returncode == 0

    _rope("--socket", custom_socket, "exit", timeout=10)


def test_rope_binding_shutdown_is_idempotent(tmp_path):
    """Regression test: Rope.shutdown() must be safe to call more than once.

    A caller that explicitly shuts down and then lets the process exit
    normally hits this twice (once explicitly, once via the atexit hook
    registered in Rope.__init__). Before the idempotency guard, the second
    call's connect() could still succeed while the server was mid-exit, but
    nothing ever serviced it -- the underlying IPC client has no receive
    timeout, so the second call hung forever instead of failing fast or
    returning. This test would hang (not fail) if the regression came back.
    """
    sys.path.insert(0, str(_project_root / "python"))
    from rope import Rope

    lib_path = _project_root / "build" / "librope.so"
    assert lib_path.is_file(), "librope.so not built"

    conf = tmp_path / "rope_shutdown.conf"
    _write_conf(conf)
    custom_socket = str(tmp_path / "shutdown.sock")

    r = Rope(lib_path=lib_path, exe_path=Path(ROPE_EXE), socket_path=custom_socket, config_path=conf)
    r.forecast(FORECAST_START, FORECAST_HORIZON)

    start = time.monotonic()
    r.shutdown()
    r.shutdown()  # must return immediately, not hang
    assert time.monotonic() - start < 5.0


# ---------------------------------------------------------------------------
# get — single point
# ---------------------------------------------------------------------------

def test_get_interp_returns_valid_point(server):
    result = server.run(
        "get", "--mode", "interp",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["density"]     > 0
    assert data["uncertainty"] >= 0


def test_get_hold_returns_valid_point(server):
    result = server.run(
        "get", "--mode", "hold",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["density"]     > 0
    assert data["uncertainty"] >= 0


def test_get_time_out_of_range_fails(server):
    result = server.run(
        "get", "--mode", "interp",
        "--time", "2030-01-01T00:00:00",
        "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
        timeout=10,
    )
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# get — batch
# ---------------------------------------------------------------------------

def test_batch_get_from_csv(server, tmp_path):
    csv = tmp_path / "queries.csv"
    csv.write_text(
        "YYYY,MM,DD,HH,MIN,SS,lst,lat,alt_km\n"
        "2024,01,01,01,00,00,12.0,45.0,400.0\n"
        "2024,01,01,01,30,00,6.0,-30.0,300.0\n"
    )
    out = tmp_path / "results.json"
    result = server.run(
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
# Idle timeout — separate short-lived server
# ---------------------------------------------------------------------------

def test_idle_timeout_shuts_down_server(tmp_path):
    # AF_UNIX path limits: 104 chars on macOS, 108 on Linux, 108 on Windows.
    # pytest embeds the full test name in tmp_path, which can exceed these
    # limits (e.g. 132 chars on macOS CI). Use tempfile.gettempdir() directly
    # so the path stays short (~25–70 chars) on all platforms.
    import tempfile
    sock = str(Path(tempfile.gettempdir()) / f"rope_idle_{os.getpid()}.sock")
    conf = tmp_path / "rope_idle.conf"
    _write_conf(conf, idle_timeout_seconds=5)

    start = _rope(
        "--socket", sock, "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", str(conf),
    )
    assert start.returncode == 0, f"idle-timeout server startup failed:\n{start.stderr}"

    # Wait long enough for the watcher to fire (timeout=5s, watcher checks every 1s,
    # so 8 seconds gives two full check cycles of margin).
    time.sleep(8)

    probe = _rope(
        "--socket", sock, "get", "--mode", "interp",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
        timeout=5,
    )
    assert probe.returncode != 0, "server should have shut down due to idle timeout"


# ---------------------------------------------------------------------------
# Pipeline load failure — server stays alive and serves other requests
# ---------------------------------------------------------------------------

def test_bad_exported_dir_forecast_fails_but_server_stays_alive(tmp_path):
    # exported_dir has no model artifacts at all, so the pipeline fails to
    # load at server startup. Per server.cpp, that failure is logged and
    # swallowed — the server keeps running and answers other requests;
    # only "forecast" should report an error.
    #
    # Socket path must stay short (AF_UNIX sun_path limit, 104 chars on
    # macOS) — pytest's tmp_path embeds the full test name and can exceed
    # that. Use tempfile.gettempdir() directly, same as the idle-timeout
    # test above.
    import tempfile
    sock      = str(Path(tempfile.gettempdir()) / f"rope_baddir_{os.getpid()}.sock")
    conf      = tmp_path / "rope_bad_dir.conf"
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    conf.write_text(
        f"[paths]\n"
        f"exported_dir = {empty_dir}\n"
        f"[server]\n"
        f"idle_timeout_seconds = 30\n"
    )

    forecast = _rope(
        "--socket", sock, "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", str(conf),
    )
    assert forecast.returncode != 0
    assert "pipeline" in forecast.stderr.lower()

    # The server itself must still be up — exit should succeed cleanly.
    exit_result = _rope("--socket", sock, "exit", timeout=10)
    assert exit_result.returncode == 0
