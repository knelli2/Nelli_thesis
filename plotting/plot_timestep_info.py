import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import h5py

# ── Path configuration ────────────────────────────────────────────────────────
SIMS_DIR = "/home/knelli/Documents/research/sims"
CASES = {
    "q1": os.path.join(SIMS_DIR, "thesis_data/lovelace_nelli_bbh_cce/CauchyData/BBH/RERUN111824_Lev2/InspiralReductionData/TimeSteps.dat"),
    "q4": os.path.join(SIMS_DIR, "thesis_data/q4_nonspinning_alex_carpenter/Full_Reductions.h5"),
}

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "images", "defense")
OUTPUT_NAME = "timestep_info.png"

# ── Plot style (matches papers-2024-spectre-first-bbh) ───────────────────────
plt.style.use("tableau-colorblind10")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.size'] = 24
COLOR_CYCLE = ['#0F2080', '#F5793A', '#85C0F9', '#A95AA1']

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data(dat_file):
    # Columns: Time, NumberOfPoints, SlabSize, MinDt, MaxDt, EffectiveDt, MinWall, MaxWall
    if os.path.splitext(dat_file)[1] in (".h5", ".hdf5"):
        with h5py.File(dat_file, "r") as f:
            for key in ("TimeSteps.dat", "TimestepInfo.dat"):
                if key in f:
                    return np.asarray(f[key])
        raise KeyError(f"No timestep dataset found in {dat_file}")
    return np.loadtxt(dat_file, comments='#')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)

for ax, (case, dat_file) in zip(axes, CASES.items()):
    data   = load_data(dat_file)
    time   = data[:, 0]
    min_dt = data[:, 3]
    max_dt = data[:, 4]
    eff_dt = data[:, 5]

    ax.semilogy(time, min_dt, color=COLOR_CYCLE[0], lw=1.5, label=r"$\Delta t_\mathrm{min}$")
    ax.semilogy(time, max_dt, color=COLOR_CYCLE[1], lw=1.5, label=r"$\Delta t_\mathrm{max}$")
    ax.semilogy(time, eff_dt, color=COLOR_CYCLE[2], lw=1.5, label=r"$\Delta t_\mathrm{eff}$")

    ax.set_ylim(1e-5, 5e-1)
    ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1])
    ax.yaxis.grid(True, which='major', color='grey', linewidth=0.7, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"$t\ [M]$")
    ax.legend(fontsize=18, ncol=3)

axes[0].set_ylabel(r"Time step $[M]$")

fig.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
fig.savefig(out_path, bbox_inches="tight", dpi=300)
print(f"Saved: {out_path}")
