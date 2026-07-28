#!/usr/bin/env python3
"""
Generate synthetic test fixtures for ROPE unit tests.

Default (no flags) — regenerates tests/fixtures/test_models/:
  base_model_00.onnx .. base_model_14.onnx  — 15 base temporal models
  meta_model.onnx                            — ensemble meta-model (inner ONNX)
  coae_decoder.onnx                          — COAE decoder (ONNX Runtime path)
  stats_ts.bin                               — feature normalizer stats (identity)
  stats_cae.bin                              — CAE denormalizer stats (identity)
  ic_table.icbin / ic_table.csv             — initial-condition lookup table
tests/fixtures/:
  sw_test.csv    — space weather stub
  ic_test.csv    — IC table stub (shared fixture reference)

With --m N:
  Generates N base models and adapts the meta model accordingly.

With --split-alt ALT (must accompany --m or standalone):
  Instead of a single coae_decoder, generates two decoder stubs:
    decoder_lo.onnx — covers alt [0, ALT), constant output 0.0 → density 1.0
    decoder_hi.onnx — covers alt [ALT, GRID_ALT), constant output 1.0 → density 10.0
    stats_cae_lo.bin, stats_cae_hi.bin — identity stats for each stage
  model_manifest.json is written with two decoder entries.

With --out-dir DIR:
  Write model artifacts to DIR instead of tests/fixtures/test_models/.
  Also writes model_manifest.json and ic_table.csv in DIR.
  Does not write sw_test.csv/ic_test.csv (those live in the shared fixtures dir).

Run locally:
    pip install onnx
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python tests/gen_fixtures.py [--m N] [--split-alt ALT] [--out-dir DIR]
"""

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT     = Path(__file__).parent
FIXTURES = ROOT / "fixtures"

GRID_LST    = 72
GRID_LAT    = 36
GRID_ALT    = 45
GRID_VOXELS = GRID_LST * GRID_LAT * GRID_ALT

DEFAULT_M = 15
K         = 10  # latent_dim
S         = 3   # seq_len
D         = 16  # K + driver_dim(6)

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--m",         type=int,  default=DEFAULT_M,
                    help=f"Number of base models (default {DEFAULT_M})")
parser.add_argument("--split-alt", type=int,  default=None,
                    help="Altitude index to split decoders at (generates two decoder stages)")
parser.add_argument("--out-dir",   type=Path, default=None,
                    help="Output directory (default: tests/fixtures/test_models/)")
args = parser.parse_args()

M         = args.m
SPLIT_ALT = args.split_alt
OUT_DIR   = args.out_dir if args.out_dir else FIXTURES / "test_models"
IS_CUSTOM = args.out_dir is not None

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIXTURES.mkdir(parents=True, exist_ok=True)

if SPLIT_ALT is not None and not (0 < SPLIT_ALT < GRID_ALT):
    print(f"ERROR: --split-alt must be in (0, {GRID_ALT}), got {SPLIT_ALT}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Binary stats files
# ---------------------------------------------------------------------------
def write_stats_bin(path, ndim, shape, mu, sigma):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", ndim))
        for s in shape:
            f.write(struct.pack("<I", s))
        for v in mu:
            f.write(struct.pack("<f", v))
        for v in sigma:
            f.write(struct.pack("<f", v))


print("Writing stats_ts.bin …")
write_stats_bin(OUT_DIR / "stats_ts.bin", ndim=1, shape=[D],
                mu=[0.0] * D, sigma=[1.0] * D)

if SPLIT_ALT is None:
    print("Writing stats_cae.bin …")
    write_stats_bin(OUT_DIR / "stats_cae.bin", ndim=1, shape=[1],
                    mu=[0.0], sigma=[1.0])
else:
    print("Writing stats_cae_lo.bin and stats_cae_hi.bin …")
    write_stats_bin(OUT_DIR / "stats_cae_lo.bin", ndim=1, shape=[1],
                    mu=[0.0], sigma=[1.0])
    write_stats_bin(OUT_DIR / "stats_cae_hi.bin", ndim=1, shape=[1],
                    mu=[0.0], sigma=[1.0])


SW_CSV = """\
datetime,f10,kp
2023-12-31T22:00:00,150.0,2.0
2023-12-31T23:00:00,150.0,2.0
2024-01-01T00:00:00,150.0,2.0
2024-01-01T01:00:00,150.0,2.0
2024-01-01T02:00:00,150.0,2.0
"""

IC_HEADER = "f10,kp," + ",".join(f"y{i+1}" for i in range(K))
IC_ZEROS  = ",".join(["0.0"] * K)
IC_CSV    = IC_HEADER + "\n"
for f10 in [100.0, 200.0]:
    for kp in [1.0, 3.0]:
        IC_CSV += f"{f10},{kp},{IC_ZEROS}\n"

if not IS_CUSTOM:
    print("Writing sw_test.csv …")
    (FIXTURES / "sw_test.csv").write_text(SW_CSV)
    print("Writing ic_test.csv …")
    (FIXTURES / "ic_test.csv").write_text(IC_CSV)

print("Writing ic_table.csv …")
(OUT_DIR / "ic_table.csv").write_text(IC_CSV)


try:
    import numpy as np
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("ERROR: onnx not installed. Run: pip install onnx numpy", file=sys.stderr)
    sys.exit(1)


def _gemm_zero(name, in_shape, out_dim):
    in_flat = int(np.prod(in_shape[1:]))
    W  = numpy_helper.from_array(np.zeros((in_flat, out_dim), dtype=np.float32), name="W")
    b  = numpy_helper.from_array(np.zeros(out_dim, dtype=np.float32),            name="b")
    rs = numpy_helper.from_array(np.array([0, in_flat], dtype=np.int64),         name="rs")
    nodes = [
        helper.make_node("Reshape", ["input", "rs"],    ["flat"]),
        helper.make_node("Gemm",    ["flat", "W", "b"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, list(in_shape))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [in_shape[0], out_dim])],
        initializer=[W, b, rs],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(m)
    return m


def _meta_inner(m_models):
    W  = numpy_helper.from_array(
        np.full((S * D, m_models), fill_value=1.0 / m_models, dtype=np.float32), name="W_meta")
    b  = numpy_helper.from_array(np.zeros(m_models, dtype=np.float32), name="b_meta")
    rs = numpy_helper.from_array(np.array([0, S * D], dtype=np.int64), name="rs_meta")
    nodes = [
        helper.make_node("Reshape", ["input", "rs_meta"],         ["flat"]),
        helper.make_node("Gemm",    ["flat", "W_meta", "b_meta"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, "meta_inner",
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, ["T", S, D])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["T", m_models])],
        initializer=[W, b, rs],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(m)
    return m


def _decoder_onnx(n_alt=GRID_ALT, fill_value=0.0):
    stage_voxels = GRID_LST * GRID_LAT * n_alt
    idx0  = numpy_helper.from_array(np.array(0,              dtype=np.int64), name="idx0")
    axes0 = numpy_helper.from_array(np.array([0],            dtype=np.int64), name="axes0")
    nvox  = numpy_helper.from_array(np.array([stage_voxels], dtype=np.int64), name="nvox")
    fill  = helper.make_tensor("fill_val", TensorProto.FLOAT, [1], [fill_value])
    nodes = [
        helper.make_node("Shape",           ["input"],               ["in_shape"]),
        helper.make_node("Gather",          ["in_shape", "idx0"],    ["batch_scalar"], axis=0),
        helper.make_node("Unsqueeze",       ["batch_scalar", "axes0"], ["batch_1d"]),
        helper.make_node("Concat",          ["batch_1d", "nvox"],    ["out_shape"], axis=0),
        helper.make_node("ConstantOfShape", ["out_shape"],           ["output"], value=fill),
    ]
    graph = helper.make_graph(
        nodes, "coae_decoder",
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, ["batch", K])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", stage_voxels])],
        initializer=[idx0, axes0, nvox],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(m)
    return m


print(f"Writing {M} base ONNX models …")
for i in range(M):
    name = f"base_model_{i:02d}.onnx"
    onnx.save(_gemm_zero(f"base_{i:02d}", in_shape=[1, S, D], out_dim=K),
              str(OUT_DIR / name))

print("Writing meta_model.onnx …")
onnx.save(_meta_inner(M), str(OUT_DIR / "meta_model.onnx"))

if SPLIT_ALT is None:
    print("Writing coae_decoder.onnx …")
    onnx.save(_decoder_onnx(), str(OUT_DIR / "coae_decoder.onnx"))
else:
    n_lo = SPLIT_ALT
    n_hi = GRID_ALT - SPLIT_ALT
    print(f"Writing decoder_lo.onnx (alt [0,{SPLIT_ALT})) …")
    onnx.save(_decoder_onnx(n_lo, fill_value=0.0), str(OUT_DIR / "decoder_lo.onnx"))
    print(f"Writing decoder_hi.onnx (alt [{SPLIT_ALT},{GRID_ALT})) …")
    onnx.save(_decoder_onnx(n_hi, fill_value=1.0), str(OUT_DIR / "decoder_hi.onnx"))


if SPLIT_ALT is None:
    try:
        import torch
        import torch.nn as nn

        class _FakeDecoder(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros(x.shape[0], 116640)

        print("Writing coae_decoder.pt …")
        scripted = torch.jit.script(_FakeDecoder())
        scripted.save(str(OUT_DIR / "coae_decoder.pt"))

    except ImportError:
        print("WARNING: torch not installed — skipping coae_decoder.pt")


def arch_for(i, m_count):
    if m_count <= 3:
        return "lstm"
    third = m_count // 3
    if i < third:
        return "lstm"
    elif i < 2 * third:
        return "gru"
    return "transformer"

if IS_CUSTOM:
    base_models = []
    for i in range(M):
        arch = arch_for(i, M)
        base_models.append({
            "file": f"base_model_{i:02d}.onnx",
            "backend": "onnx",
            "architecture": arch,
            "inter_op_threads": 1,
        })

    if SPLIT_ALT is None:
        decoders = [{
            "backends": {"onnx": "coae_decoder.onnx", "libtorch": "coae_decoder.pt"},
            "stats": "stats_cae.bin",
            "alt_start": 0,
            "alt_end": GRID_ALT,
        }]
    else:
        decoders = [
            {
                "backends": {"onnx": "decoder_lo.onnx"},
                "stats": "stats_cae_lo.bin",
                "alt_start": 0,
                "alt_end": SPLIT_ALT,
            },
            {
                "backends": {"onnx": "decoder_hi.onnx"},
                "stats": "stats_cae_hi.bin",
                "alt_start": SPLIT_ALT,
                "alt_end": GRID_ALT,
            },
        ]

    has_libtorch = any(
        "libtorch" in d.get("backends", {}) for d in decoders
    )
    runtime_reqs: dict = {"onnxruntime": "1.25"}
    if has_libtorch:
        runtime_reqs["libtorch"] = "2.7"

    manifest = {
        "schema_version": 1,
        "kind": "stacked_ensemble",
        "runtime_requirements": runtime_reqs,
        "latent_dim": K,
        "drivers": {
            "source": "celestrak_sw",
            "columns": [
                {"name": "f10", "description": "F10.7 cm solar radio flux (SFU)."},
                {"name": "kp", "description": "Kp planetary geomagnetic index (0-9 scale)."},
                {"name": "t1", "description": "sin(2*pi*hour/24) - diurnal phase harmonic."},
                {"name": "t2", "description": "cos(2*pi*hour/24) - diurnal phase harmonic."},
                {"name": "t3", "description": "sin(2*pi*day_of_year/365.25) - annual phase harmonic."},
                {"name": "t4", "description": "cos(2*pi*day_of_year/365.25) - annual phase harmonic."},
            ],
        },
        "validated": False,
        "grid": {
            "n_lst": GRID_LST, "n_lat": GRID_LAT, "n_alt": GRID_ALT,
            "lat_min_deg": -87.5, "lat_max_deg": 87.5,
            "alt_min_km": 100.0, "alt_max_km": 980.0,
        },
        "ic": {
            "kind": "ic_lookup_table",
            "params": {"grid_axes": ["f10", "kp"], "file": "ic_table.icbin"},
        },
        "stacked_ensemble": {
            "seq_len": S,
            "decode_batch_size": 120,
            "base_models": base_models,
            "meta_model": {"file": "meta_model.onnx", "backend": "onnx"},
            "decoders": decoders,
        },
    }
    print("Writing model_manifest.json …")
    (OUT_DIR / "model_manifest.json").write_text(json.dumps(manifest, indent=2))

print(f"Done.  Fixtures written to {OUT_DIR}")
