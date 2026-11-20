"""Plot TGV Re=1600 dissipation-rate curves from verification/results/tgv_re1600_*.csv."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tgv import DNS_PEAK_EPS, DNS_PEAK_T  # noqa: E402

res = Path(__file__).parent / "results"
fig, ax = plt.subplots(figsize=(6.5, 4))
for f in sorted(res.glob("tgv_re1600_n*_*.csv")):
    d = np.loadtxt(f, delimiter=",", skiprows=1)
    label = f.stem.replace("tgv_re1600_", "").replace("_roe", "")
    ax.plot(d[:, 0], d[:, 1], label=label, lw=1.4, ls="-" if "n64" in label else "--")
ax.plot([DNS_PEAK_T], [DNS_PEAK_EPS], "k*", ms=12, label="DNS peak (van Rees 2011)")
ax.set_xlabel("t"); ax.set_ylabel(r"$\varepsilon = -dk/dt$"); ax.set_xlim(0, 20); ax.grid(alpha=0.3)
ax.legend(fontsize=8); ax.set_title("Taylor-Green vortex, Re = 1600, M = 0.1")
fig.tight_layout(); fig.savefig(res / "tgv_re1600_dissipation.png", dpi=130)
print("wrote", res / "tgv_re1600_dissipation.png")
