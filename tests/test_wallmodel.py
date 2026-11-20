"""Wall models: Warp kernel vs numpy reference, and a-priori accuracy against Lee & Moser DNS."""

from pathlib import Path

import numpy as np
import pytest
import warp as wp

from wmles_wrap.config import config_from_dict
from wmles_wrap.numerics.kernels import WM_ALGEBRAIC, WM_EQUILIBRIUM, make_kernels
from wmles_wrap.physics.gas import GasModel
from wmles_wrap.precision import get_precision
from wmles_wrap.wallmodel.reference import WallModelParams, algebraic_model, equilibrium_model

LM = Path(__file__).resolve().parents[1] / "cases" / "channel" / "LM_Channel_5200_mean_prof.dat"


def _gas(mach=0.3, re=1e5, visc="sutherland"):
    cfg = config_from_dict({"grid": {"file": "-"}, "flow": {"mach": mach, "reynolds": re, "temperature": 300.0},
                            "gas": {"viscosity": visc}})
    return GasModel.from_config(cfg)


def _params(K, gas, model, prm):
    P = K["Params"]()
    P.gamma, P.R, P.cv, P.cp = gas.gamma, gas.R, gas.cv, gas.cp
    P.prandtl, P.prandtl_t = gas.prandtl, gas.prandtl_t
    P.visc_model = {"sutherland": 0, "power": 1, "constant": 2}[gas.viscosity]
    P.s_nd, P.power_exp, P.inv_re = gas.s_nd, gas.power_exponent, 1.0 / gas.reynolds
    P.wm_model, P.wm_kappa, P.wm_aplus = model, prm.kappa, prm.a_plus
    P.wm_n, P.wm_r, P.wm_iters = prm.n_points, prm.stretching, prm.iterations
    P.wm_iters_warm = prm.iterations_warm
    return P


def _run_kernel(gas, model, U, T_e, p_e, side, cell_index, iso, T_w, dy=0.02, h_dist=0.0, prm=None):
    """Column of identical cells normal to a y-wall; returns tau_w vector, Q, u_tau."""
    prec = get_precision("float64")
    K = make_kernels(prec)
    prm = prm or WallModelParams()
    ng, nc, nt = 2, 10, 2
    shape = (nt + 2 * ng, nc + 2 * ng, nt + 2 * ng)
    W = np.zeros(shape + (5,))
    W[..., 0] = p_e / (gas.R * T_e)
    W[..., 1] = U * 0.8
    W[..., 3] = U * 0.6
    W[..., 2] = 0.01  # wall-normal component must be ignored
    W[..., 4] = p_e
    xc = np.zeros(shape + (3,))
    j = np.arange(shape[1])
    yc = (j - ng + 0.5) * dy  # wall at y = 0 (low side); high side wall at y = nc dy
    xc[..., 1] = yc[None, :, None]
    tw = wp.zeros((nt, nt), dtype=prec.vec3)
    q = wp.zeros((nt, nt), dtype=wp.float64)
    ut = wp.zeros((nt, nt), dtype=wp.float64)
    wp.launch(K["wall_model"], dim=(nt, nt),
              inputs=[wp.array(W, dtype=prec.vec5), wp.array(xc, dtype=prec.vec3), 1, side, ng, nc, cell_index,
                      h_dist, iso, T_w, 0, _params(K, gas, model, prm)],
              outputs=[tw, q, ut])
    return tw.numpy()[0, 0], q.numpy()[0, 0], ut.numpy()[0, 0]


@pytest.mark.parametrize("model", [WM_EQUILIBRIUM, WM_ALGEBRAIC])
@pytest.mark.parametrize("iso, T_w", [(1, 1.0), (1, 1.3), (0, 1.0)])
@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("U, cell_index", [(0.5, 1), (1.0, 3), (2.0, 5)])
def test_kernel_matches_reference(model, iso, T_w, side, U, cell_index):
    gas = _gas()
    T_e, p_e, dy = 1.1, gas.R * 1.05, 0.02
    tw, q, ut = _run_kernel(gas, model, U, T_e, p_e, side, cell_index, iso, T_w, dy)
    h = (cell_index - 0.5) * dy
    Tw_ref = T_w if iso else None
    if model == WM_EQUILIBRIUM:
        tau_ref, q_ref = equilibrium_model(gas, U, h, T_e, p_e, Tw_ref)
    else:
        tau_ref, q_ref = algebraic_model(gas, U, h, T_e, p_e, Tw_ref)
    assert np.linalg.norm(tw) == pytest.approx(tau_ref, rel=1e-10)
    assert tw / np.linalg.norm(tw) == pytest.approx([0.8, 0.0, 0.6], abs=1e-12)  # along tangential velocity
    assert q == pytest.approx(q_ref, rel=1e-9, abs=1e-14)
    if not iso:
        assert q == 0.0


def test_exchange_by_distance():
    gas = _gas()
    a = _run_kernel(gas, WM_EQUILIBRIUM, 1.0, 1.0, gas.R, 0, 3, 1, 1.0)[0]
    b = _run_kernel(gas, WM_EQUILIBRIUM, 1.0, 1.0, gas.R, 0, 1, 1, 1.0, h_dist=0.045)[0]  # 3rd centre at 0.05
    assert np.allclose(a, b, rtol=1e-12)


@pytest.mark.skipif(not LM.exists(), reason="Lee & Moser data not present")
@pytest.mark.parametrize("h, tol", [(0.05, 0.005), (0.1, 0.01), (0.2, 0.015)])
def test_equilibrium_model_a_priori_lee_moser(h, tol):
    """Fed with the DNS mean velocity at y = h, the equilibrium model recovers the DNS friction
    velocity at Re_tau = 5200 (low Mach, constant viscosity)."""
    d = np.loadtxt(LM, comments="%")
    ub_plus = np.trapezoid(d[:, 2], d[:, 0]) / d[-1, 0]
    utau = 1.0 / ub_plus
    gas = _gas(mach=0.1, re=d[-1, 1] / d[-1, 0] * ub_plus, visc="constant")
    U = np.interp(h, d[:, 0], d[:, 2]) * utau
    tau, _ = equilibrium_model(gas, U, h, 1.0, gas.R, 1.0)
    assert np.sqrt(tau) == pytest.approx(utau, rel=tol)


def test_warm_start_converges():
    """Warm start from a slightly different tau_w converges to the cold-start solution."""
    gas = _gas()
    prm = WallModelParams()
    tau, _ = equilibrium_model(gas, 0.75, 0.0667, 1.0, gas.R, 1.0, prm)
    tau_w, _ = equilibrium_model(gas, 0.75, 0.0667, 1.0, gas.R, 1.0, prm, tau0=1.03 * tau)
    assert tau_w == pytest.approx(tau, rel=2e-3)


@pytest.mark.parametrize("model", ["equilibrium_ode", "algebraic"])
def test_wall_traction_balances_forcing(tmp_path, model):
    """With the wall model updated once per step, the momentum removed by the modeled wall
    traction over a step must equal the mass-flux controller's correction (exactly, since the
    periodic interior fluxes cancel and SSP-RK3 is consistent)."""
    from wmles_wrap.io.plot3d import write_plot3d
    from wmles_wrap.solver import Solver

    y = np.linspace(-1, 1, 13)
    x = np.stack(np.meshgrid(np.linspace(0, 2, 9), y, np.linspace(0, 1, 7), indexing="ij"), axis=-1)
    write_plot3d(tmp_path / "ch.p3d", [x])
    cfg = config_from_dict(
        {
            "grid": {"file": "ch.p3d"},
            "flow": {"mach": 0.2, "reynolds": 1e5, "temperature": 300.0},
            "wall_model": {"type": model, "exchange": {"cell_index": 2}},
            "boundaries": [{"faces": ["jmin", "jmax"], "type": "wall_modeled", "thermal": "isothermal",
                            "T_wall": 1.0}],
            "periodic": [{"translation": [2, 0, 0]}, {"translation": [0, 0, 1]}],
            "forcing": {"type": "constant_mass_flux"},
            "init": {"type": "channel", "amplitude": 0.1},
            "parallel": {"devices": ["cpu"], "partitions": 2},
        },
        case_dir=tmp_path,
    )
    s = Solver(cfg)
    vol = sum(s.interior(s.metrics[b.id].vol.numpy()).sum() for b in s.subs)
    for _ in range(3):
        s.advance(nsteps=1)
        # traction used during the step (computed at its first stage, stored on the faces)
        drag = 0.0
        for sid, d in s.integrator.data.items():
            m = s.metrics[sid]
            ng = s.ng
            for (ax, side), f in d.wm.items():
                Sj = m.Sj.numpy()[ng:-ng, ng if side == 0 else ng + d.sub.nc[1], ng:-ng]
                drag += (f.tw.numpy()[..., 0] * np.linalg.norm(Sj, axis=-1)).sum()
        assert s.integrator.forcing_history[-1][1] == pytest.approx(drag / vol, rel=1e-10)
    assert 0.03 < s.integrator.wall_friction_velocity() < 0.06


@pytest.mark.parametrize("mach, T_w", [(2.0, 1.5), (3.0, 1.0), (0.8, 2.0)])
def test_equilibrium_model_crocco_busemann(mach, T_w):
    """With Pr = Pr_t = 1 the equilibrium energy equation reduces exactly to the Crocco-Busemann
    relation cp T + u^2/2 = linear in u; checks the compressible energy coupling of the model."""
    cfg = config_from_dict({"grid": {"file": "-"}, "flow": {"mach": mach, "reynolds": 1e5, "temperature": 300.0},
                            "gas": {"prandtl": 1.0, "prandtl_t": 1.0}})
    gas = GasModel.from_config(cfg)
    U, h, T_e = 1.0, 0.05, 1.0
    prm = WallModelParams(n_points=64, stretching=1.08, iterations=60)
    tau, Q, y, u, T = equilibrium_model(gas, U, h, T_e, gas.R, T_w, prm, return_profiles=True)
    T_cb = T_w + (T_e - T_w + U**2 / (2 * gas.cp)) * u / U - u**2 / (2 * gas.cp)
    assert np.abs(T - T_cb).max() < 2e-3 * max(abs(T_w - T_e), U**2 / (2 * gas.cp))
    # wall heat flux consistent with the slope dT/du at the wall
    # Q = k_w dT/dy|_w = (cp/Pr) tau_w dT/du|_0  (mu du/dy = tau_w at the wall, Pr = 1)
    assert Q == pytest.approx(gas.cp * tau * (T_e - T_w + U**2 / (2 * gas.cp)) / U, rel=5e-3)
