from pathlib import Path

import numpy as np
import pytest

from wmles_wrap.io.extrude import channel_nk, extrude_block
from wmles_wrap.io.plot3d import read_plot3d, write_plot3d

CHANNEL = Path(__file__).resolve().parents[1] / "cases" / "channel"


@pytest.mark.parametrize("name", ["channel_ar1_1dwm_129x33.p3d", "channel_ar2_1dwm_65x33.p3d"])
def test_channel_matches_plot3d_package(name):
    plot3d = pytest.importorskip("plot3d")
    ours = read_plot3d(CHANNEL / name)
    ref = plot3d.read_plot3D(str(CHANNEL / name), binary=True)
    assert len(ours) == len(ref) == 1
    for a, r in zip(ours, ref):
        assert np.array_equal(a[..., 0], r.X)
        assert np.array_equal(a[..., 1], r.Y)
        assert np.array_equal(a[..., 2], r.Z)


def test_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    blocks = [rng.random((5, 4, 3, 3)), rng.random((2, 7, 2, 3))]
    write_plot3d(tmp_path / "g.p3d", blocks)
    back = read_plot3d(tmp_path / "g.p3d")
    assert all(np.array_equal(a, b) for a, b in zip(blocks, back))


def _fortran_record(arr):
    data = arr.tobytes()
    m = np.array([len(data)], dtype=arr.dtype.byteorder.replace("=", "<") + "i4").tobytes()
    return m + data + m


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("real", ["f4", "f8"])
@pytest.mark.parametrize("fortran", [True, False])
@pytest.mark.parametrize("multiblock", [True, False])
def test_format_variants(tmp_path, endian, real, fortran, multiblock):
    rng = np.random.default_rng(1)
    x = rng.random((4, 3, 2, 3)).astype(np.float32).astype(np.float64)
    parts = []
    if multiblock:
        parts.append(np.array([1], dtype=f"{endian}i4"))
    parts.append(np.array(x.shape[:3], dtype=f"{endian}i4"))
    parts.append(np.ascontiguousarray(x.transpose(3, 2, 1, 0)).astype(f"{endian}{real}"))
    path = tmp_path / "v.p3d"
    with open(path, "wb") as fh:
        for p in parts:
            fh.write(_fortran_record(p) if fortran else p.tobytes())
    back = read_plot3d(path)
    assert len(back) == 1 and np.array_equal(back[0], x)


def test_extrude_matches_fortran_rule():
    b = read_plot3d(CHANNEL / "channel_ar1_1dwm_129x33.p3d")[0]
    assert channel_nk(129, 1) == 49 and channel_nk(65, 2) == 49
    e = extrude_block(b, 49, 4.8)
    assert e.shape == (129, 33, 49, 3)
    assert np.allclose(e[..., 2][0, 0], np.linspace(0, 4.8, 49))
    assert np.array_equal(e[:, :, 7, :2], b[:, :, 0, :2])
