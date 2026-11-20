"""Reference (numpy) implementations of the wall models. The Warp kernels in
numerics/kernels.py mirror these line by line; tests compare the two.

Equilibrium ODE wall model (compressible; e.g. Kawai & Larsson 2012, Bocquet et al. 2012):
    d/dy [ (mu + mu_t) du/dy ] = 0
    d/dy [ (mu + mu_t) u du/dy + cp (mu/Pr + mu_t/Pr_t) dT/dy ] = 0
    mu_t = kappa rho y u_tau* [1 - exp(-y*/A+)]^2,  u_tau* = sqrt(tau_w/rho),  y* = y sqrt(rho tau_w)/mu
with u(0) = 0, u(h) = U_e, T(0) = T_w (isothermal) or dT/dy(0) = 0 (adiabatic), T(h) = T_e,
p constant across the layer (rho = p / (R T)). Both equations integrate once exactly:
    (mu + mu_t) du/dy = tau_w           ->  tau_w = U_e / int_0^h dy / (mu + mu_t)
    tau_w u + k_eff dT/dy = Q           ->  Q = k dT/dy|_w (heat flux convention: n into the fluid)
so each iteration needs only quadratures on a stretched 1D grid (one GPU thread per wall face).

Algebraic model: Reichardt's law of the wall for u_tau (Newton iterations, wall properties) and the
Crocco-Busemann relation for the wall heat flux.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..physics.gas import GasModel


@dataclass
class WallModelParams:
    kappa: float = 0.41
    a_plus: float = 17.0
    n_points: int = 40
    stretching: float = 1.15
    iterations: int = 30  # cold start (from the algebraic model)
    iterations_warm: int = 4  # warm start (from the previous tau_w)


def wm_grid(h: float, n: int, r: float) -> np.ndarray:
    if abs(r - 1.0) < 1e-12:
        return np.linspace(0.0, h, n)
    dy0 = h * (r - 1.0) / (r ** (n - 1) - 1.0)
    return np.concatenate([[0.0], np.cumsum(dy0 * r ** np.arange(n - 1))])


def reichardt(yp):
    return np.log(1.0 + 0.4 * yp) / 0.4 + 7.8 * (1.0 - np.exp(-yp / 11.0) - yp / 11.0 * np.exp(-yp / 3.0))


def reichardt_deriv(yp):
    return 1.0 / (1.0 + 0.4 * yp) + 7.8 * (np.exp(-yp / 11.0) / 11.0 - np.exp(-yp / 3.0) / 11.0
                                           + yp / 33.0 * np.exp(-yp / 3.0))


def algebraic_utau(U, h, nu, iters=30):
    """Solve U/u_tau = f(h u_tau / nu) (Reichardt) for u_tau with Newton's method."""
    ut = max(np.sqrt(nu * U / h), 1e-12)  # laminar guess
    for _ in range(iters):
        yp = h * ut / nu
        g = ut * reichardt(yp) - U
        dg = reichardt(yp) + yp * reichardt_deriv(yp)
        ut = max(ut - g / dg, 0.1 * ut)
    return ut


def algebraic_model(gas: GasModel, U, h, T_e, p, T_w=None):
    """Returns tau_w, Q (= k dT/dn at the wall). T_w None -> adiabatic wall."""
    Tw = T_e if T_w is None else T_w
    rho_w = p / (gas.R * Tw)
    mu_w = float(gas.mu(np.asarray(Tw)))
    ut = algebraic_utau(U, h, mu_w / rho_w)
    tau = rho_w * ut * ut
    if T_w is None or U <= 0:
        return tau, 0.0
    r = gas.prandtl ** (1.0 / 3.0)
    T_aw = T_e + r * U * U / (2.0 * gas.cp)
    return tau, gas.cp / gas.prandtl * (T_aw - T_w) * tau / U


def equilibrium_model(gas: GasModel, U, h, T_e, p, T_w=None, prm: WallModelParams | None = None,
                      return_profiles=False, tau0=None):
    """Returns tau_w, Q (= k dT/dn at the wall, n into the fluid). T_w None -> adiabatic wall.
    tau0: previous tau_w for a warm start (prm.iterations_warm iterations)."""
    prm = prm or WallModelParams()
    y = wm_grid(h, prm.n_points, prm.stretching)
    dy = np.diff(y)
    iso = T_w is not None
    T = T_w + (T_e - T_w) * y / h if iso else np.full_like(y, T_e)
    if tau0 is not None and tau0 > 0:
        tau, iters = tau0, prm.iterations_warm
    else:
        tau, iters = algebraic_model(gas, U, h, T_e, p, T_w)[0], prm.iterations
    Q = 0.0
    u = U * y / h
    for _ in range(iters):
        rho = p / (gas.R * T)
        mu = gas.mu(T)
        ystar = y * np.sqrt(rho * tau) / mu
        mut = prm.kappa * rho * y * np.sqrt(tau / rho) * (1.0 - np.exp(-ystar / prm.a_plus)) ** 2
        f = 1.0 / (mu + mut)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dy)])
        tau_new = U / cum[-1]
        u = U * cum / cum[-1]
        keff = gas.cp * (mu / gas.prandtl + mut / gas.prandtl_t)
        g = 1.0 / keff
        gu = tau_new * u / keff
        Ig = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * dy)])
        Igu = np.concatenate([[0.0], np.cumsum(0.5 * (gu[1:] + gu[:-1]) * dy)])
        if iso:
            Q = (T_e - T_w + Igu[-1]) / Ig[-1]
            T = T_w + Q * Ig - Igu
        else:
            Q = 0.0
            T = T_e + (Igu[-1] - Igu)
        tau = tau_new
    if return_profiles:
        return tau, Q, y, u, T
    return tau, Q
