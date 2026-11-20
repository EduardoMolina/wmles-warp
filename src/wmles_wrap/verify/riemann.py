"""Exact solution of the 1D Riemann problem for an ideal gas (Toro, ch. 4)."""

from __future__ import annotations

import numpy as np


def _f(p, rho_k, p_k, gamma):
    a_k = np.sqrt(gamma * p_k / rho_k)
    if p > p_k:  # shock
        A = 2.0 / ((gamma + 1.0) * rho_k)
        B = (gamma - 1.0) / (gamma + 1.0) * p_k
        f = (p - p_k) * np.sqrt(A / (p + B))
        df = np.sqrt(A / (B + p)) * (1.0 - 0.5 * (p - p_k) / (B + p))
    else:  # rarefaction
        f = 2.0 * a_k / (gamma - 1.0) * ((p / p_k) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        df = 1.0 / (rho_k * a_k) * (p / p_k) ** (-(gamma + 1.0) / (2.0 * gamma))
    return f, df


def exact_riemann(left, right, gamma, x, t, x0=0.5):
    """left/right = (rho, u, p). Returns rho, u, p arrays at positions x and time t."""
    rl, ul, pl = left
    rr, ur, pr = right
    al, ar = np.sqrt(gamma * pl / rl), np.sqrt(gamma * pr / rr)
    p = max(1e-8, 0.5 * (pl + pr))
    for _ in range(100):
        fl, dfl = _f(p, rl, pl, gamma)
        fr, dfr = _f(p, rr, pr, gamma)
        dp = (fl + fr + ur - ul) / (dfl + dfr)
        p = max(1e-10, p - dp)
        if abs(dp) < 1e-14 * p:
            break
    fl, _ = _f(p, rl, pl, gamma)
    fr, _ = _f(p, rr, pr, gamma)
    us = 0.5 * (ul + ur) + 0.5 * (fr - fl)
    g1, g2 = (gamma - 1.0) / (gamma + 1.0), (gamma - 1.0) / (2.0 * gamma)

    s = (np.asarray(x, dtype=np.float64) - x0) / max(t, 1e-300)
    rho = np.empty_like(s)
    u = np.empty_like(s)
    pp = np.empty_like(s)
    for n, si in enumerate(s):
        if si <= us:  # left of contact
            if p > pl:
                rs = rl * (p / pl + g1) / (g1 * p / pl + 1.0)
                sh = ul - al * np.sqrt((gamma + 1) / (2 * gamma) * p / pl + g2)
                st = (rl, ul, pl) if si < sh else (rs, us, p)
            else:
                rs = rl * (p / pl) ** (1.0 / gamma)
                ash = al * (p / pl) ** g2
                if si < ul - al:
                    st = (rl, ul, pl)
                elif si > us - ash:
                    st = (rs, us, p)
                else:
                    uf = 2.0 / (gamma + 1) * (al + (gamma - 1) / 2 * ul + si)
                    rf = rl * (2 / (gamma + 1) + (gamma - 1) / ((gamma + 1) * al) * (ul - si)) ** (2 / (gamma - 1))
                    pf = pl * (2 / (gamma + 1) + (gamma - 1) / ((gamma + 1) * al) * (ul - si)) ** (2 * gamma / (gamma - 1))
                    st = (rf, uf, pf)
        else:
            if p > pr:
                rs = rr * (p / pr + g1) / (g1 * p / pr + 1.0)
                sh = ur + ar * np.sqrt((gamma + 1) / (2 * gamma) * p / pr + g2)
                st = (rr, ur, pr) if si > sh else (rs, us, p)
            else:
                rs = rr * (p / pr) ** (1.0 / gamma)
                ast = ar * (p / pr) ** g2
                if si > ur + ar:
                    st = (rr, ur, pr)
                elif si < us + ast:
                    st = (rs, us, p)
                else:
                    uf = 2.0 / (gamma + 1) * (-ar + (gamma - 1) / 2 * ur + si)
                    rf = rr * (2 / (gamma + 1) - (gamma - 1) / ((gamma + 1) * ar) * (ur - si)) ** (2 / (gamma - 1))
                    pf = pr * (2 / (gamma + 1) - (gamma - 1) / ((gamma + 1) * ar) * (ur - si)) ** (2 * gamma / (gamma - 1))
                    st = (rf, uf, pf)
        rho[n], u[n], pp[n] = st
    return rho, u, pp
