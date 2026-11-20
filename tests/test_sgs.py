"""SGS eddy-viscosity kernels vs numpy reference implementations."""

import numpy as np
import pytest
import warp as wp

from wmles_wrap.numerics.kernels import SGS_SMAGORINSKY, SGS_VREMAN, SGS_WALE, make_kernels
from wmles_wrap.precision import get_precision


def _ref_nut(model, g, delta, c):
    if model == SGS_VREMAN:
        a = g.T  # alpha_ij = d u_j / d x_i
        b = delta**2 * a.T @ a
        B = b[0, 0] * b[1, 1] - b[0, 1] ** 2 + b[0, 0] * b[2, 2] - b[0, 2] ** 2 + b[1, 1] * b[2, 2] - b[1, 2] ** 2
        return c * np.sqrt(B / np.sum(a * a)) if B > 0 else 0.0
    S = 0.5 * (g + g.T)
    if model == SGS_WALE:
        g2 = g @ g
        Sd = 0.5 * (g2 + g2.T) - np.trace(g2) / 3 * np.eye(3)
        ss, sd = np.sum(S * S), np.sum(Sd * Sd)
        return (c * delta) ** 2 * sd**1.5 / (ss**2.5 + sd**1.25)
    return (c * delta) ** 2 * np.sqrt(2 * np.sum(S * S))


@pytest.mark.parametrize("model, c", [(SGS_VREMAN, 0.07), (SGS_WALE, 0.5), (SGS_SMAGORINSKY, 0.17)])
def test_sgs_kernels(model, c):
    prec = get_precision("float64")
    K = make_kernels(prec)
    rng = np.random.default_rng(model)
    n = 6
    G = rng.standard_normal((n, n, n, 3, 3))
    G[0, 0, 0] = 0.0  # zero gradient -> zero viscosity
    G[1, 0, 0] = [[0, 1.0, 0], [0, 0, 0], [0, 0, 0]]  # pure shear
    vol = rng.uniform(0.5, 2.0, (n, n, n))
    rho = rng.uniform(0.8, 1.2, (n, n, n))
    W = np.zeros((n, n, n, 5))
    W[..., 0] = rho
    P = K["Params"]()
    P.sgs_model = model
    P.sgs_c = c
    mut = wp.zeros((n, n, n), dtype=wp.float64)
    wp.launch(K["sgs"], dim=(n - 2,) * 3,
              inputs=[wp.array(W, dtype=prec.vec5), wp.array(G, dtype=K["mat33"]), wp.array(vol, dtype=wp.float64), P],
              outputs=[mut])
    got = mut.numpy()
    for i, j, k in np.ndindex(n - 2, n - 2, n - 2):
        I = (i + 1, j + 1, k + 1)
        ref = rho[I] * _ref_nut(model, G[I], vol[I] ** (1 / 3), c)
        assert got[I] == pytest.approx(ref, rel=1e-12, abs=1e-300)


def test_vreman_and_wale_vanish_in_pure_shear():
    """Both models are designed to give nu_t = 0 for pure shear (laminar near-wall flow)."""
    g = np.array([[0, 1.0, 0], [0, 0, 0], [0, 0, 0]])
    assert _ref_nut(SGS_VREMAN, g, 1.0, 0.07) == 0.0
    assert _ref_nut(SGS_WALE, g, 1.0, 0.5) == 0.0
    assert _ref_nut(SGS_SMAGORINSKY, g, 1.0, 0.17) > 0.0
