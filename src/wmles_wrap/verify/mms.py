"""Method of manufactured solutions for the compressible Navier-Stokes equations.

Manufactured primitive fields (rho, u, v, w, T) are sums of trigonometric terms with exact
time derivatives, gradients and Hessians. The source term

    S = dU/dt + div(F_inv(U)) - div(F_visc(U, grad U))

is assembled analytically (no symbolic algebra needed), consistent with the solver's
nondimensional equations: p = rho R T, E = cv T + |u|^2/2, tau = mu (G + G^T - 2/3 div I),
q = -k grad T with k = mu cp / Pr, mu = mu(T) / Re (Sutherland, power law or constant).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..physics.gas import GasModel


@dataclass
class TrigField:
    """phi = c0 + sum_m amp_m * sin(k_m . x + omega_m t + phase_m)."""

    c0: float
    terms: list[tuple[float, tuple[float, float, float], float, float]] = field(default_factory=list)

    def _args(self, x, t):
        for amp, k, om, ph in self.terms:
            k = np.asarray(k, dtype=np.float64)
            yield amp, k, om, x @ k + om * t + ph

    def value(self, x, t):
        out = np.full(x.shape[:-1], self.c0, dtype=np.float64)
        for amp, _, _, a in self._args(x, t):
            out += amp * np.sin(a)
        return out

    def dt(self, x, t):
        out = np.zeros(x.shape[:-1])
        for amp, _, om, a in self._args(x, t):
            out += amp * om * np.cos(a)
        return out

    def grad(self, x, t):
        out = np.zeros(x.shape)
        for amp, k, _, a in self._args(x, t):
            out += (amp * np.cos(a))[..., None] * k
        return out

    def hess(self, x, t):
        out = np.zeros(x.shape + (3,))
        for amp, k, _, a in self._args(x, t):
            out -= (amp * np.sin(a))[..., None, None] * np.outer(k, k)
        return out


class ManufacturedSolution:
    """With ``nut`` (and ``wall_distance``, a function of x), the solution includes the SA working
    variable: mu_t = rho nut f_v1 enters the mean-flow viscous terms, and ``sa_source`` is the SA
    forcing (numerics/sa.py). Gradients of mu_t and of the SA diffusion coefficient are central
    differences of the analytic functions (step ``fd_step``)."""

    def __init__(self, gas: GasModel, rho: TrigField, u: TrigField, v: TrigField, w: TrigField, T: TrigField,
                 viscous: bool = True, nut: TrigField | None = None, wall_distance=None, fd_step: float = 1e-5):
        self.gas = gas
        self.f = {"rho": rho, "u": u, "v": v, "w": w, "T": T}
        self.viscous = viscous
        self.nut = nut
        self.wall_distance = wall_distance
        self.fd_step = fd_step

    def primitives(self, x, t):
        """[rho, u, v, w, p] at points x (..., 3)."""
        rho = self.f["rho"].value(x, t)
        T = self.f["T"].value(x, t)
        vel = [self.f[c].value(x, t) for c in "uvw"]
        return np.stack([rho, *vel, rho * self.gas.R * T], axis=-1)

    def _mu(self, T):
        g = self.gas
        mu = g.mu(T)
        if g.viscosity == "sutherland":
            dmu = mu * (1.5 / T - 1.0 / (T + g.s_nd))
        elif g.viscosity == "power":
            dmu = mu * g.power_exponent / T
        else:
            dmu = np.zeros_like(T)
        return mu, dmu

    def source(self, x, t):
        g = self.gas
        R, cv, cp = g.R, g.cv, g.cp
        fr, fT = self.f["rho"], self.f["T"]
        rho, rho_t, grho = fr.value(x, t), fr.dt(x, t), fr.grad(x, t)
        T, T_t, gT = fT.value(x, t), fT.dt(x, t), fT.grad(x, t)
        lapT = np.trace(fT.hess(x, t), axis1=-2, axis2=-1)
        fu = [self.f[c] for c in "uvw"]
        u = np.stack([f.value(x, t) for f in fu], axis=-1)
        u_t = np.stack([f.dt(x, t) for f in fu], axis=-1)
        G = np.stack([f.grad(x, t) for f in fu], axis=-2)  # G[..., i, j] = d u_i / d x_j
        H = np.stack([f.hess(x, t) for f in fu], axis=-3)  # H[..., i, j, k] = d2 u_i / dx_j dx_k
        div = np.trace(G, axis1=-2, axis2=-1)

        p = rho * R * T
        gp = R * (T[..., None] * grho + rho[..., None] * gT)
        p_t = R * (T * rho_t + rho * T_t)
        E = cv * T + 0.5 * np.einsum("...i,...i", u, u)
        gE = cv * gT + np.einsum("...i,...ij->...j", u, G)
        rhoE = rho * E

        # time derivatives of conservative variables
        S = np.zeros(x.shape[:-1] + (5,))
        S[..., 0] = rho_t
        S[..., 1:4] = rho_t[..., None] * u + rho[..., None] * u_t
        S[..., 4] = rho_t * E + rho * (cv * T_t + np.einsum("...i,...i", u, u_t))

        # inviscid flux divergence
        div_rhou = np.einsum("...i,...i", u, grho) + rho * div
        S[..., 0] += div_rhou
        S[..., 1:4] += u * div_rhou[..., None] + rho[..., None] * np.einsum("...j,...ij->...i", u, G) + gp
        grhoE_p = E[..., None] * grho + rho[..., None] * gE + gp
        S[..., 4] += (rhoE + p) * div + np.einsum("...i,...i", u, grhoE_p)
        del p_t

        if self.viscous:
            mu, dmu = self._mu(T)
            gmu = dmu[..., None] * gT
            mu_lam, gmu_lam = mu, gmu
            if self.nut is not None:
                mut = self.eddy_viscosity(x, t)
                gmut = self._fd_grad(lambda y: self.eddy_viscosity(y, t), x)
                mu, gmu = mu + mut, gmu + gmut
            eye = np.eye(3)
            sym = G + np.swapaxes(G, -1, -2) - (2.0 / 3.0) * div[..., None, None] * eye
            tau = mu[..., None, None] * sym
            lap_u = np.einsum("...ijj->...i", H)
            grad_div = np.einsum("...jji->...i", H)  # d/dx_i (d u_j / d x_j)
            div_tau = np.einsum("...j,...ij->...i", gmu, sym) + mu[..., None] * (lap_u + grad_div / 3.0)
            div_tau_u = np.einsum("...i,...i", div_tau, u) + np.einsum("...ij,...ij", tau, G)
            k = mu * cp / g.prandtl
            gk = gmu * cp / g.prandtl
            if self.nut is not None:
                k = cp * (mu_lam / g.prandtl + mut / g.prandtl_t)
                gk = cp * (gmu_lam / g.prandtl + gmut / g.prandtl_t)
            div_q = -(np.einsum("...i,...i", gk, gT) + k * lapT)
            S[..., 1:4] -= div_tau
            S[..., 4] += -div_tau_u + div_q
        return S


    # ---- SA working variable ----------------------------------------------------------------
    def _fd_grad(self, fn, x):
        h = self.fd_step
        out = np.zeros(x.shape)
        for j in range(3):
            e = np.zeros(3)
            e[j] = h
            out[..., j] = (fn(x + e) - fn(x - e)) / (2 * h)
        return out

    def nu(self, x, t):
        return self.gas.mu(self.f["T"].value(x, t)) / self.f["rho"].value(x, t)

    def eddy_viscosity(self, x, t):
        from ..numerics.sa import CV1

        nt = self.nut.value(x, t)
        chi = nt / self.nu(x, t)
        fv1 = chi**3 / (chi**3 + CV1**3)
        return np.where(nt > 0, self.f["rho"].value(x, t) * nt * fv1, 0.0)

    def sa_source(self, x, t):
        """Forcing of the nut equation (per unit volume): dnut/dt + u.grad nut - (P - D) - diffusion."""
        from ..numerics.sa import CB2, SIGMA, sa_source_numpy

        fn = self.nut
        nt, gnt = fn.value(x, t), fn.grad(x, t)
        lapnt = np.trace(fn.hess(x, t), axis1=-2, axis2=-1)
        u = np.stack([self.f[c].value(x, t) for c in "uvw"], axis=-1)
        G = np.stack([self.f[c].grad(x, t) for c in "uvw"], axis=-2)
        om = np.linalg.norm(np.stack([G[..., 2, 1] - G[..., 1, 2], G[..., 0, 2] - G[..., 2, 0],
                                      G[..., 1, 0] - G[..., 0, 1]], -1), axis=-1)
        nu = self.nu(x, t)

        def coef(y):  # nu + nut f_n
            n = fn.value(y, t)
            v = self.nu(y, t)
            c3 = (n / v) ** 3
            return v + np.where(n >= 0, n, n * (16.0 + c3) / (16.0 - c3))

        c = coef(x)
        gc = self._fd_grad(coef, x)
        diff = (np.einsum("...i,...i", gc, gnt) + c * lapnt + CB2 * np.einsum("...i,...i", gnt, gnt)) / SIGMA
        src = sa_source_numpy(nt, om, nu, self.wall_distance(x))
        return fn.dt(x, t) + np.einsum("...i,...i", u, gnt) - src - diff


def default_solution(gas: GasModel, viscous: bool = True, L: float = 1.0) -> ManufacturedSolution:
    """Smooth, fully 3D, unsteady solution on a domain of size ~L (all terms active)."""
    k = 2.0 * np.pi / L
    return ManufacturedSolution(
        gas,
        rho=TrigField(1.0, [(0.10, (k, 0.5 * k, 0.0), 1.0, 0.3), (0.05, (0.0, k, 0.5 * k), -0.7, 1.1)]),
        u=TrigField(0.8, [(0.10, (0.5 * k, k, 0.5 * k), 0.8, 0.0)]),
        v=TrigField(0.2, [(0.08, (k, 0.0, 0.5 * k), -0.5, 0.7)]),
        w=TrigField(-0.1, [(0.06, (0.5 * k, 0.5 * k, k), 0.6, 1.9)]),
        T=TrigField(1.0, [(0.08, (0.5 * k, k, 0.0), 0.9, 2.3), (0.04, (k, 0.0, k), 0.0, 0.4)]),
        viscous=viscous,
    )


def sa_solution(gas: GasModel, L: float = 1.0) -> ManufacturedSolution:
    """Steady solution with the SA variable (nut = nu_inf (8 + 3 sin...), chi ~ 5..11, so mu_t,
    production, destruction and diffusion are all active); wall distance d = 0.05 + y + 0.2 z."""
    k = 2.0 * np.pi / L
    nu_inf = float(gas.mu(np.float64(1.0)))
    return ManufacturedSolution(
        gas,
        rho=TrigField(1.0, [(0.08, (k, 0.5 * k, 0.0), 0.0, 0.3)]),
        u=TrigField(0.8, [(0.10, (0.5 * k, k, 0.5 * k), 0.0, 0.0)]),
        v=TrigField(0.1, [(0.08, (k, 0.0, 0.5 * k), 0.0, 0.7)]),
        w=TrigField(-0.05, [(0.06, (0.5 * k, 0.5 * k, k), 0.0, 1.9)]),
        T=TrigField(1.0, [(0.06, (0.5 * k, k, 0.0), 0.0, 2.3)]),
        nut=TrigField(8.0 * nu_inf, [(3.0 * nu_inf, (k, 0.5 * k, 0.5 * k), 0.0, 0.9)]),
        wall_distance=lambda x: 0.05 + x[..., 1] + 0.2 * x[..., 2],
    )
