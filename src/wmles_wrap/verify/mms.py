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
    def __init__(self, gas: GasModel, rho: TrigField, u: TrigField, v: TrigField, w: TrigField, T: TrigField,
                 viscous: bool = True):
        self.gas = gas
        self.f = {"rho": rho, "u": u, "v": v, "w": w, "T": T}
        self.viscous = viscous

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
            eye = np.eye(3)
            sym = G + np.swapaxes(G, -1, -2) - (2.0 / 3.0) * div[..., None, None] * eye
            tau = mu[..., None, None] * sym
            lap_u = np.einsum("...ijj->...i", H)
            grad_div = np.einsum("...jji->...i", H)  # d/dx_i (d u_j / d x_j)
            div_tau = np.einsum("...j,...ij->...i", gmu, sym) + mu[..., None] * (lap_u + grad_div / 3.0)
            div_tau_u = np.einsum("...i,...i", div_tau, u) + np.einsum("...ij,...ij", tau, G)
            k = mu * cp / g.prandtl
            gk = gmu * cp / g.prandtl
            div_q = -(np.einsum("...i,...i", gk, gT) + k * lapT)
            S[..., 1:4] -= div_tau
            S[..., 4] += -div_tau_u + div_q
        return S


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
