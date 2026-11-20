"""Calorically perfect gas in nondimensional form (see config.py for the scaling).

Conservative state vector: U = [rho, rho*u, rho*v, rho*w, rho*E],  E = cv*T + |u|^2/2,
p = rho*R*T with R = 1/(gamma*Mach^2), cv = R/(gamma-1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CaseConfig


@dataclass(frozen=True)
class GasModel:
    gamma: float
    mach: float
    reynolds: float
    prandtl: float
    prandtl_t: float
    viscosity: str
    s_nd: float  # Sutherland constant / T_ref
    power_exponent: float

    @classmethod
    def from_config(cls, cfg: CaseConfig) -> "GasModel":
        return cls(
            gamma=cfg.gas.gamma,
            mach=cfg.flow.mach,
            reynolds=cfg.flow.reynolds,
            prandtl=cfg.gas.prandtl,
            prandtl_t=cfg.gas.prandtl_t,
            viscosity=cfg.gas.viscosity,
            s_nd=cfg.sutherland_s_nd,
            power_exponent=cfg.gas.power_exponent,
        )

    @property
    def R(self) -> float:
        return 1.0 / (self.gamma * self.mach**2)

    @property
    def cv(self) -> float:
        return self.R / (self.gamma - 1.0)

    @property
    def cp(self) -> float:
        return self.gamma * self.cv

    def mu(self, T):
        """Nondimensional dynamic viscosity (includes 1/Re)."""
        if self.viscosity == "sutherland":
            m = T**1.5 * (1.0 + self.s_nd) / (T + self.s_nd)
        elif self.viscosity == "power":
            m = T**self.power_exponent
        else:
            m = np.ones_like(T)
        return m / self.reynolds

    # ---- numpy conversions (host side: initialization, output) ----------------------------
    def conservative(self, rho, u, v, w, T):
        E = self.cv * T + 0.5 * (u * u + v * v + w * w)
        return np.stack([rho, rho * u, rho * v, rho * w, rho * E], axis=-1)

    def primitives(self, U: np.ndarray) -> dict[str, np.ndarray]:
        rho = U[..., 0]
        u, v, w = U[..., 1] / rho, U[..., 2] / rho, U[..., 3] / rho
        ke = 0.5 * (u * u + v * v + w * w)
        T = (U[..., 4] / rho - ke) / self.cv
        p = rho * self.R * T
        a = np.sqrt(self.gamma * self.R * np.maximum(T, 0.0))
        return {
            "rho": rho, "u": u, "v": v, "w": w, "p": p, "T": T,
            "mach": np.sqrt(2 * ke) / a,
        }

    def freestream(self, alpha_deg: float = 0.0, beta_deg: float = 0.0):
        a, b = np.radians(alpha_deg), np.radians(beta_deg)
        return 1.0, np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b), 1.0
