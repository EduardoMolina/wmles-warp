"""Mean-velocity and Reynolds-stress profiles of a channel run from its stats.npz.

Assumes a single-block channel with x streamwise (i), y wall-normal (j), z spanwise (k), walls
at y = -1 and y = +1. Averages over i, k and the two channel halves; scales with the friction
velocity from the mean wall shear (du/dy at the wall from the first cell) and, as a check,
from the mean driving force of the mass-flux controller (tau_w = F h).

    conda run -n pyvista env PYTHONPATH=src python verification/channel_profiles.py out/stats.npz \\
        --reynolds 2800 --mach 0.2 [--plot] [--ref ref.dat]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wmles_wrap.config import config_from_dict
from wmles_wrap.physics.gas import GasModel

KAPPA, B = 0.41, 5.2


def profiles(stats_path: str | Path, reynolds: float, mach: float, temperature: float = 300.0):
    f = np.load(stats_path)
    names = list(f["fields"])
    a = f["block0000"]
    xc = f["xc0000"]
    F = {k: a[..., i].mean(axis=(0, 2)) for i, k in enumerate(names)}
    y = xc[..., 1].mean(axis=(0, 2))
    n = len(y)
    h = n // 2

    def fold(q, sign=1.0):  # average lower and (mirrored) upper half
        return 0.5 * (q[:h] + sign * q[::-1][:h])

    eta = 1.0 + fold(y, -1.0)  # distance from the wall
    gas = GasModel.from_config(config_from_dict(
        {"grid": {"file": "-"}, "flow": {"mach": mach, "reynolds": reynolds, "temperature": temperature}}))
    T_w = fold(F["T"])[0]  # first cell ~ wall temperature for isothermal walls
    mu_w = float(gas.mu(np.array(T_w)))
    u = fold(F["u"])
    rho = fold(F["rho"])
    tau_w = mu_w * u[0] / eta[0]
    rho_w = rho[0]
    utau = np.sqrt(tau_w / rho_w)
    nu_w = mu_w / rho_w
    out = {
        "y+": eta * utau / nu_w,
        "u+": u / utau,
        "urms+": np.sqrt(np.maximum(fold(F["uu"]) - u**2, 0)) / utau,
        "vrms+": np.sqrt(np.maximum(fold(F["vv"]) - fold(F["v"], -1.0) ** 2, 0)) / utau,
        "wrms+": np.sqrt(np.maximum(fold(F["ww"]) - fold(F["w"]) ** 2, 0)) / utau,
        "uv+": -fold(F["uv"] - F["u"] * F["v"], -1.0) / utau**2,
        "mut/mu": fold(F["mut"]) / mu_w,
    }
    forcing = f["forcing"]
    utau_force = np.sqrt(np.mean(forcing[len(forcing) // 2:, 1]) / rho_w) if len(forcing) else np.nan
    return out, {"utau": utau, "Re_tau": utau / nu_w, "utau_from_forcing": utau_force,
                 "samples": int(f["count"]), "Ub/utau": 1.0 / utau}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("--reynolds", type=float, required=True)
    ap.add_argument("--mach", type=float, required=True)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--ref", help="optional reference profile file with columns y+ u+ (e.g. DNS)")
    args = ap.parse_args()
    prof, info = profiles(args.stats, args.reynolds, args.mach)
    print("  ".join(f"{k} {v:.4g}" for k, v in info.items()))
    print(f"{'y+':>8} {'u+':>8} {'urms+':>7} {'vrms+':>7} {'wrms+':>7} {'-uv+':>7} {'mut/mu':>7}")
    for i in range(len(prof["y+"])):
        print("".join(f"{prof[k][i]:8.3f}" for k in ("y+", "u+", "urms+", "vrms+", "wrms+", "uv+", "mut/mu")))
    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        yp = prof["y+"]
        ax[0].semilogx(yp, prof["u+"], "o-", ms=3, label="wmles_wrap")
        yy = np.logspace(0, np.log10(yp.max()), 100)
        ax[0].semilogx(yy, np.minimum(yy, np.log(yy) / KAPPA + B), "k--", lw=1, label="y+ / log law")
        if args.ref:
            r = np.loadtxt(args.ref)
            ax[0].semilogx(r[:, 0], r[:, 1], "k-", lw=1, label="reference")
        ax[0].set_xlabel("y+"); ax[0].set_ylabel("u+"); ax[0].legend(); ax[0].grid(alpha=0.3)
        for k in ("urms+", "vrms+", "wrms+", "uv+"):
            ax[1].plot(yp, prof[k], label=k)
        ax[1].set_xlabel("y+"); ax[1].legend(); ax[1].grid(alpha=0.3)
        out = Path(args.stats).with_name("channel_profiles.png")
        fig.tight_layout(); fig.savefig(out, dpi=130)
        print("wrote", out)


if __name__ == "__main__":
    main()
