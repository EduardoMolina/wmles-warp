"""Surface pressure and skin friction of an airfoil WMLES run, from its stats.npz (and forces.dat).

Uses the time-averaged wall samples written for every wall-modeled face (``wall{b}_{face}``:
first-cell pressure, modeled traction tau_w * t, heat flux; ``wallxf``/``wallS``: face centroids
and area vectors pointing into the fluid). Averages over the span (k), then

    Cp = (p - p_inf) / (1/2),  p_inf = 1 / (gamma M^2)    (rho_inf = |u_inf| = 1)
    Cf = 2 tau_w . t,  t = unit surface tangent pointing from the leading toward the trailing edge

and integrates CL, CD (pressure + friction) from the mean surface fields. With --forces, also
averages the CL/CD history of forces.dat over the statistics window and plots it.

    conda run -n pyvista env PYTHONPATH=src python verification/airfoil_surface.py \\
        cases/a_airfoil/out/stats.npz --case cases/a_airfoil/case.yaml \\
        --forces cases/a_airfoil/out/forces.dat --ref "surface data.dat" --plot

Reference (A-airfoil, alpha = 13.3 deg, Re_c = 1e7, M = 0.15): Tamaki & Kawai, AIAA J. 61(2), 2023,
WRLES database https://www.klab.mech.tohoku.ac.jp/database/2022TamakiAIAAJ/ ("surface data.dat":
x/c, Cp, Cf, H, Re_theta, Re_delta*).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wmles_wrap.config import load_config


def surface(stats_path: str | Path, mach: float, gamma: float = 1.4) -> list[dict]:
    """One dict per wall-modeled parent face, ordered along the face: label (upper/lower from the
    mean normal), x, y (span-averaged face centroids), Cp, Cf, total and pressure force."""
    z = np.load(stats_path)
    fields = list(z["wall_fields"])
    p_inf = 1.0 / (gamma * mach**2)
    out = []
    for key in sorted(k[4:] for k in z.files if k.startswith("wall") and k[4:5].isdigit()):
        a, xf, S = z[f"wall{key}"], z[f"wallxf{key}"], z[f"wallS{key}"]
        face = key.split("_")[1]
        span_axis = 1 if face[0] in "ij" else None  # (i or j, k) faces: k is the 2nd tangential axis
        if span_axis is None:
            raise ValueError(f"face {key}: expected an i or j face with k spanwise")
        F = {f: a[..., n].mean(axis=span_axis) for n, f in enumerate(fields)}
        x = xf.mean(axis=span_axis)
        tw = np.stack([F["tw_x"], F["tw_y"], F["tw_z"]], -1)
        t = np.gradient(x, axis=0)
        t /= np.linalg.norm(t, axis=-1, keepdims=True)
        if t[np.argmax(x[:, 0]), 0] < 0:  # orient towards the trailing edge (max x)
            t = -t
        out.append({
            "key": key, "label": f"{'upper' if S[..., 1].sum() > 0 else 'lower'} ({key})", "x": x[:, 0], "y": x[:, 1], "Cp": 2.0 * (F["p"] - p_inf),
            "Cf": 2.0 * np.einsum("ij,ij->i", tw, t),
            "force": (np.linalg.norm(S, axis=-1)[..., None] * a[..., 1:4]
                      - (a[..., 0] - p_inf)[..., None] * S).reshape(-1, 3).sum(0),
            "pressure_force": (-(a[..., 0] - p_inf)[..., None] * S).reshape(-1, 3).sum(0),
        })
    return out


def coefficients(F: np.ndarray, alpha_deg: float, ref_area: float) -> tuple[float, float]:
    a = np.radians(alpha_deg)
    q = 0.5 * ref_area
    return float(F @ [-np.sin(a), np.cos(a), 0.0]) / q, float(F @ [np.cos(a), np.sin(a), 0.0]) / q


def read_numeric_table(path: str | Path, min_cols: int = 3) -> np.ndarray:
    """Rows of a whitespace/comma table that parse as >= min_cols floats (headers skipped)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            v = [float(s) for s in line.replace(",", " ").split()]
        except ValueError:
            continue
        if len(v) >= min_cols:
            rows.append(v[:min(len(v), 6)])
    return np.array(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stats")
    ap.add_argument("--case", required=True, help="case YAML (Mach, gamma, alpha, ref_area, stats window)")
    ap.add_argument("--forces", help="forces.dat (CL/CD history)")
    ap.add_argument("--ref", help="reference surface data (x/c, Cp, Cf, ...)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default=None, help="figure path (default: next to stats.npz)")
    args = ap.parse_args()

    cfg = load_config(args.case)
    ref_area = cfg.flow.ref_area or 1.0
    faces = surface(args.stats, cfg.flow.mach, cfg.gas.gamma)
    z = np.load(args.stats)
    print(f"statistics: {int(z['count'])} samples, last at step {int(z['step'])}, t = {float(z['time']):.4f}")
    Ftot = sum(f["force"] for f in faces)
    Fp = sum(f["pressure_force"] for f in faces)
    cl, cd = coefficients(Ftot, cfg.flow.alpha, ref_area)
    clp, cdp = coefficients(Fp, cfg.flow.alpha, ref_area)
    print(f"mean surface fields: CL = {cl:.4f}  CD = {cd:.5f}  (pressure: CL {clp:.4f}, CD {cdp:.5f}; "
          f"friction CD {cd - cdp:.5f})")
    for f in faces:
        sep = f["x"][f["Cf"] < 0]
        msg = f"x/c in [{sep.min():.3f}, {sep.max():.3f}]" if sep.size else "none"
        print(f"  {f['label']}: {len(f['x'])} points, Cp min {f['Cp'].min():.3f}, reversed flow (Cf < 0): {msg}")

    hist = None
    if args.forces:
        hist = np.loadtxt(args.forces)
        t0 = cfg.run.stats_start_time if cfg.run.stats_start_time is not None else hist[0, 1]
        w = hist[:, 1] >= t0
        if w.any():
            print(f"forces.dat, t >= {t0:g} ({w.sum()} samples): CL = {hist[w, 5].mean():.4f} "
                  f"(rms {hist[w, 5].std():.4f})  CD = {hist[w, 6].mean():.5f} (rms {hist[w, 6].std():.5f})")

    ref = read_numeric_table(args.ref) if args.ref else None

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = 3 if hist is not None else 2
        fig, ax = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
        for f in faces:
            ax[0].plot(f["x"], f["Cp"], "-", lw=1.2, label=f"WMLES {f['label']}")
            ax[1].plot(f["x"], f["Cf"], "-", lw=1.2, label=f"WMLES {f['label']}")
        if ref is not None:
            ax[0].plot(ref[:, 0], ref[:, 1], "k.", ms=2, label="WRLES (Tamaki & Kawai)")
            ax[1].plot(ref[:, 0], ref[:, 2], "k.", ms=2, label="WRLES (Tamaki & Kawai)")
        ax[0].invert_yaxis()
        ax[0].set(xlabel="x/c", ylabel="$C_p$", title="Mean surface pressure")
        ax[1].set(xlabel="x/c", ylabel="$C_f$", title="Mean skin friction (+: towards TE)")
        ax[1].axhline(0, color="0.6", lw=0.6)
        for a in ax[:2]:
            a.legend(fontsize=7)
        if hist is not None:
            ax[2].plot(hist[:, 1], hist[:, 5], label="$C_L$")
            ax[2].plot(hist[:, 1], 10 * hist[:, 6], label="$10\\,C_D$")
            if cfg.run.stats_start_time is not None:
                ax[2].axvspan(cfg.run.stats_start_time, hist[-1, 1], color="0.9", label="averaging")
            ax[2].set(xlabel="t (CTU)", title="Force history")
            ax[2].legend(fontsize=7)
        fig.tight_layout()
        out = Path(args.out) if args.out else Path(args.stats).with_name("airfoil_surface.png")
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
