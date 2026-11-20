"""Decaying compressible isotropic turbulence with shocklets (Johnsen et al., JCP 2010, sec. 4.3).

Initial solenoidal velocity with spectrum E(k) ~ k^4 exp(-2 k^2 / k0^2), k0 = 4, on [0, 2 pi]^3,
uniform rho = p-consistent T = 1, turbulent Mach number Mt = <u.u>^(1/2) / <c> = 0.6,
Re_lambda = <rho> u' lambda0 / mu = 100 with u' = (<u.u>/3)^(1/2), lambda0 = 2 / k0,
mu ~ T^0.76, Pr = 0.7. Time in eddy turnover times tau = lambda0 / u'.

Reports kinetic energy, enstrophy, dilatation variance, T rms and the fraction of cells where
the shock sensor is active, for grid-convergence/robustness checks.

    conda run -n pyvista env PYTHONPATH=src python verification/isotropic.py --n 64 --t-end 4
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from wmles_wrap.config import config_from_dict
from wmles_wrap.io.plot3d import write_plot3d
from wmles_wrap.parallel.devices import init_warp
from wmles_wrap.solver import Solver

K0, MT, RE_LAMBDA = 4.0, 0.6, 100.0


def solenoidal_field(n: int, k0: float = K0, seed: int = 1) -> np.ndarray:
    """Random solenoidal velocity (n, n, n, 3) on cell centres of [0, 2pi]^3, <u.u> = 1."""
    rng = np.random.default_rng(seed)
    k1 = np.fft.fftfreq(n, 1.0 / n)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    kk = np.stack([kx, ky, kz], -1)
    kmag = np.sqrt((kk**2).sum(-1))
    uh = rng.standard_normal((n, n, n, 3)) + 1j * rng.standard_normal((n, n, n, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        uh -= kk * ((kk * uh).sum(-1) / np.where(kmag > 0, kmag**2, 1.0))[..., None]
        spec = kmag**4 * np.exp(-2.0 * kmag**2 / k0**2)
        # shell energy of the random field ~ k^2 |uh|^2; target E(k) -> amplitude ~ sqrt(E / k^2)
        amp = np.where(kmag > 0, np.sqrt(spec) / np.maximum(kmag, 1e-12), 0.0)
    uh *= amp[..., None]
    uh[kmag > n / 3] = 0.0  # dealias / keep resolvable
    u = np.real(np.fft.ifftn(uh, axes=(0, 1, 2)))
    # shift to cell centres (grid nodes at 2pi i / n, centres at +pi/n): phase shift in Fourier space
    shift = np.exp(1j * (kx + ky + kz)[..., None] * np.pi / n)
    u = np.real(np.fft.ifftn(uh * shift, axes=(0, 1, 2)))
    u -= u.mean(axis=(0, 1, 2))
    return u / np.sqrt((u**2).sum(-1).mean())


def make_isotropic(n=32, dissipation="roe", sensor="ducros_pressure", sgs="none", partitions=1,
                   precision="float64", seed=1):
    tmp = Path(tempfile.mkdtemp())
    L = 2 * np.pi
    x = np.stack(np.meshgrid(*(np.linspace(0, L, n + 1),) * 3, indexing="ij"), axis=-1)
    write_plot3d(tmp / "box.p3d", [x])
    urms3 = MT  # <u.u>^(1/2) with c = 1 (flow.mach = 1, T = 1)
    uprime = urms3 / np.sqrt(3.0)
    lam0 = 2.0 / K0
    mu = uprime * lam0 / RE_LAMBDA
    cfg = config_from_dict(
        {
            "grid": {"file": "box.p3d"},
            "flow": {"mach": 1.0, "reynolds": 1.0 / mu, "temperature": 300.0},
            "gas": {"viscosity": "power", "power_exponent": 0.76, "prandtl": 0.7},
            "numerics": {"dissipation": dissipation, "sensor": sensor, "precision": precision, "cfl": 0.8},
            "sgs": {"model": sgs},
            "periodic": [{"translation": t} for t in ([L, 0, 0], [0, L, 0], [0, 0, L])],
            "parallel": {"devices": ["cpu"], "partitions": partitions},
        },
        case_dir=tmp,
    )
    s = Solver(cfg)
    u = solenoidal_field(n, seed=seed) * urms3
    R = s.gas.R
    ng = s.ng
    for sb in s.subs:
        W = np.zeros(sb.padded_shape_cells(ng) + (5,))
        lo, hi = sb.lo, sb.hi
        Wi = W[ng:-ng, ng:-ng, ng:-ng]
        Wi[..., 0] = 1.0
        Wi[..., 1:4] = u[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        Wi[..., 4] = R
        Ti = np.ones(Wi.shape[:-1])
        W[ng:-ng, ng:-ng, ng:-ng] = Wi
        U = np.zeros_like(W)
        U[ng:-ng, ng:-ng, ng:-ng] = s.gas.conservative(Wi[..., 0], Wi[..., 1], Wi[..., 2], Wi[..., 3], Ti)
        s.set_state(sb.id, U)
    tau = lam0 / uprime
    return s, tau


def diagnostics(s) -> dict[str, float]:
    I = s.integrator
    I.rhs(s.time)  # refresh W, gradients, sensor
    ke = ens = dil = t2 = vol = active = ncell = 0.0
    tmean = 0.0
    for sid, d in I.data.items():
        g = s.ng
        W = s.interior(d.W.numpy()).astype(np.float64)
        GU = s.interior(d.GU.numpy()).astype(np.float64)
        th = s.interior(d.th.numpy())
        v = s.interior(s.metrics[sid].vol.numpy()).astype(np.float64)
        rho = W[..., 0]
        T = W[..., 4] / (rho * s.gas.R)
        ke += (0.5 * rho * (W[..., 1:4] ** 2).sum(-1) * v).sum()
        w = np.stack([GU[..., 2, 1] - GU[..., 1, 2], GU[..., 0, 2] - GU[..., 2, 0], GU[..., 1, 0] - GU[..., 0, 1]], -1)
        ens += ((w**2).sum(-1) * v).sum()
        dil += ((GU[..., 0, 0] + GU[..., 1, 1] + GU[..., 2, 2]) ** 2 * v).sum()
        tmean += (T * v).sum()
        t2 += (T**2 * v).sum()
        vol += v.sum()
        active += (th > 0.01).sum()
        ncell += th.size
        del g
    tm = tmean / vol
    return {"k": ke / vol, "enstrophy": ens / vol, "dilatation2": dil / vol,
            "Trms": np.sqrt(max(t2 / vol - tm**2, 0.0)), "sensor_active": active / ncell}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--t-end", type=float, default=4.0, help="in eddy turnover times")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--sgs", default="none")
    ap.add_argument("--sensor", default="ducros_pressure")
    args = ap.parse_args()
    init_warp()
    s, tau = make_isotropic(args.n, sgs=args.sgs, sensor=args.sensor)
    d0 = diagnostics(s)
    print(f"Mt={MT} Re_lambda={RE_LAMBDA} n={args.n} sgs={args.sgs} sensor={args.sensor}  tau={tau:.4f}")
    print(f"{'t/tau':>6} {'k/k0':>8} {'enstr/e0':>9} {'dil2/enstr':>11} {'Trms':>9} {'sensor%':>8} {'steps':>6}")
    rows = []
    for tt in np.linspace(0, args.t_end, args.samples + 1):
        if tt > 0:
            s.advance(t_end=tt * tau)
        d = diagnostics(s)
        rows.append((tt, d))
        print(f"{tt:6.2f} {d['k'] / d0['k']:8.4f} {d['enstrophy'] / d0['enstrophy']:9.4f} "
              f"{d['dilatation2'] / max(d['enstrophy'], 1e-30):11.3e} {d['Trms']:9.2e} "
              f"{100 * d['sensor_active']:8.2f} {s.step:6d}")
    out = Path(__file__).parent / "results" / f"isotropic_mt06_n{args.n}_{args.sgs}_{args.sensor}.csv"
    out.parent.mkdir(exist_ok=True)
    np.savetxt(out, [[t, d["k"], d["enstrophy"], d["dilatation2"], d["Trms"], d["sensor_active"]] for t, d in rows],
               delimiter=",", header="t_over_tau,k,enstrophy,dilatation2,Trms,sensor_active", comments="")
    print("wrote", out)


if __name__ == "__main__":
    main()
