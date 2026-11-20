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


def profiles(stats_path: str | Path, reynolds: float, mach: float, temperature: float = 300.0,
             utau_source: str = "wall"):
    """utau_source: 'wall' (resolved du/dy at the first cell; wall-resolved LES) or 'forcing'
    (mean driving force of the mass-flux controller = mean wall shear; use for WMLES)."""
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
    if utau_source == "forcing":
        scale = utau / utau_force
        utau = utau_force
        for k in ("u+", "urms+", "vrms+", "wrms+"):
            out[k] = out[k] * scale
        out["uv+"] = out["uv+"] * scale**2
        out["y+"] = out["y+"] / scale
    return out, {"utau": utau, "Re_tau": utau / nu_w, "utau_from_forcing": utau_force,
                 "samples": int(f["count"]), "Ub/utau": 1.0 / utau}


def lee_moser(mean_file: str | Path, fluc_file: str | Path | None = None) -> dict[str, np.ndarray]:
    """Lee & Moser (2015) DNS profiles (y+, U+, rms+, -uv+) and U_b+ from their .dat files."""
    m = np.loadtxt(mean_file, comments="%")
    out = {"y+": m[:, 1], "u+": m[:, 2], "Ub+": np.trapezoid(m[:, 2], m[:, 0]) / m[-1, 0]}
    if fluc_file is not None and Path(fluc_file).exists():
        fl = np.loadtxt(fluc_file, comments="%")
        rms = np.sqrt(np.maximum(fl[:, 2:5], 0.0))  # clip round-off negatives at the wall
        out.update({"urms+": rms[:, 0], "vrms+": rms[:, 1], "wrms+": rms[:, 2], "uv+": -fl[:, 5]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("--reynolds", type=float, required=True)
    ap.add_argument("--mach", type=float, required=True)
    ap.add_argument("--utau", choices=("wall", "forcing"), default="wall",
                    help="friction velocity source: resolved wall gradient (WRLES) or forcing (WMLES)")
    ap.add_argument("--lm", help="Lee & Moser mean-profile .dat (the matching _vel_fluc_ file is used if present)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--label", default="wmles_wrap")
    args = ap.parse_args()
    prof, info = profiles(args.stats, args.reynolds, args.mach, utau_source=args.utau)
    ref = None
    if args.lm:
        ref = lee_moser(args.lm, args.lm.replace("mean_prof", "vel_fluc_prof"))
        info["utau_DNS"] = 1.0 / ref["Ub+"]
        info["utau_error_%"] = 100.0 * (info["utau"] * ref["Ub+"] - 1.0)
    print("  ".join(f"{k} {v:.4g}" for k, v in info.items()))
    print(f"{'y+':>8} {'u+':>8} {'urms+':>7} {'vrms+':>7} {'wrms+':>7} {'-uv+':>7} {'mut/mu':>7}")
    for i in range(len(prof["y+"])):
        print("".join(f"{prof[k][i]:8.3f}" for k in ("y+", "u+", "urms+", "vrms+", "wrms+", "uv+", "mut/mu")))
    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        yp = prof["y+"]
        ax[0].semilogx(yp, prof["u+"], "o-", ms=3, label=args.label)
        if ref is not None:
            ax[0].semilogx(ref["y+"][1:], ref["u+"][1:], "k-", lw=1, label="DNS (Lee & Moser 2015)")
        else:
            yy = np.logspace(0, np.log10(yp.max()), 100)
            ax[0].semilogx(yy, np.minimum(yy, np.log(yy) / KAPPA + B), "k--", lw=1, label="y+ / log law")
        ax[0].set_xlabel("y+"); ax[0].set_ylabel("u+"); ax[0].legend(); ax[0].grid(alpha=0.3)
        for k, c in zip(("urms+", "vrms+", "wrms+", "uv+"), ("C0", "C1", "C2", "C3")):
            ax[1].semilogx(yp, prof[k], "o-", ms=3, color=c, label=k)
            if ref is not None and k in ref:
                ax[1].semilogx(ref["y+"][1:], ref[k][1:], "-", lw=1, color=c, alpha=0.6)
        ax[1].set_xlabel("y+"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        if ref is not None:
            ax[1].set_title("lines: DNS", fontsize=9)
        out = Path(args.stats).with_name("channel_profiles.png")
        fig.tight_layout(); fig.savefig(out, dpi=130)
        print("wrote", out)


if __name__ == "__main__":
    main()
