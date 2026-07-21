"""Zarr export round-trip: `rope forecast --zarr` vs the binary cache file."""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

zarr = pytest.importorskip("zarr")
xr = pytest.importorskip("xarray")
np = pytest.importorskip("numpy")

_project_root = Path(__file__).parent.parent.parent
_default_exe  = _project_root / "build" / ("rope.exe" if sys.platform == "win32" else "rope")

ROPE_EXE    = os.environ.get("ROPE_EXE", str(_default_exe))
FIXTURE_DIR = Path(os.environ.get("ROPE_FIXTURE_DIR", _project_root / "tests" / "fixtures"))

FORECAST_START   = "2024-01-01 00:00:00"
FORECAST_HORIZON = 3


def _write_conf(path: Path) -> None:
    exported_dir = FIXTURE_DIR / "test_models"
    path.write_text(
        f"[paths]\n"
        f"exported_dir = {exported_dir}\n"
        f"driver_path  = {FIXTURE_DIR / 'test_models' / 'sw_test.swbin'}\n"
    )


@pytest.fixture(scope="module")
def zarr_export(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rope_zarr")
    conf = tmp / "rope.conf"
    _write_conf(conf)
    container = tmp / "zarr_out"

    result = subprocess.run(
        [ROPE_EXE, "--cache-path", str(tmp / "forecast_grid.bin"), "forecast",
         "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
         "--config", str(conf), "--zarr", str(container)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0 and "without Zarr support" in result.stderr:
        pytest.skip("build compiled without Zarr support")
    assert result.returncode == 0, f"forecast --zarr failed:\n{result.stderr}"
    data = json.loads(result.stdout.strip().splitlines()[-1])

    return {"cache": str(tmp / "forecast_grid.bin"), "store": data["zarr_path"]}


def test_zarr_store_matches_binary_cache(zarr_export):
    ds = xr.open_dataset(zarr_export["store"], engine="zarr")

    with open(zarr_export["cache"], "rb") as f:
        _, _, n_lst, n_lat, n_alt, H = struct.unpack("<IIiiii", f.read(24))
        f.read(32 + 4)  # lat/alt ranges, reserved
        voxels = n_lst * n_lat * n_alt
        times = np.frombuffer(f.read(8 * H), dtype="<i8")
        density = np.frombuffer(f.read(4 * H * voxels), dtype="<f4").reshape(H, n_lst, n_lat, n_alt)
        uncertainty = np.frombuffer(f.read(4 * H * voxels), dtype="<f4").reshape(H, n_lst, n_lat, n_alt)

    assert (ds.sizes["time"], ds.sizes["lst"], ds.sizes["lat"], ds.sizes["alt"]) == (H, n_lst, n_lat, n_alt)
    assert np.array_equal(density, ds["density"].values)
    assert np.array_equal(uncertainty, ds["uncertainty"].values)
    assert np.array_equal(times, ds["time"].values.astype("datetime64[s]").astype("int64"))
    assert not np.isnan(ds["lst"].values).any()
    assert not np.isnan(ds["latent"].values).any()


def test_zarr_global_attrs_present(zarr_export):
    ds = xr.open_dataset(zarr_export["store"], engine="zarr")
    assert ds.attrs["Conventions"] == "CF-1.8"
    assert ds.attrs["model_kind"] == "stacked_ensemble"
    assert len(ds.attrs["model_manifest_hash"]) == 64
    assert "rope_version" in ds.attrs
