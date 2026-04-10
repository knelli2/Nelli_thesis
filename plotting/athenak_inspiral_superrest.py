"""
Plot time series of selected spin-weighted spherical harmonic modes
from rhOverM_Strain_N192.h5 (AthenaK inspiral, superrest frame).
Columns in each dataset: [time, real, imag].
"""

import os

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use("tableau-colorblind10")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.size'] = 24

COLOR_CYCLE = ['#0F2080', '#F5793A', '#85C0F9', '#A95AA1']

H5_PATH = "/home/knelli/Documents/research/sims/thesis_data/modes_comp/rhOverM_Strain_N192.h5"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "images", "cross_code_cce")

MODES = [
    (2, 2, "Y_l2_m2.dat"),
    (2, 0, "Y_l2_m0.dat"),
    (3, 2, "Y_l3_m2.dat"),
    (4, 4, "Y_l4_m4.dat"),
]

fig, axes = plt.subplots(len(MODES), 1, figsize=(12, 14), sharex=True)

with h5py.File(H5_PATH, "r") as f:
    # Find peak of |h_22| to use as t=0
    d22 = f["Y_l2_m2.dat"][()]
    amp22 = np.sqrt(d22[:, 1]**2 + d22[:, 2]**2)
    t_peak = d22[np.argmax(amp22), 0]

    for ax, (color, (l, m, key)) in zip(axes, zip(COLOR_CYCLE, MODES)):
        data = f[key][()]           # shape (N, 3): [time, real, imag]
        t  = data[:, 0] - t_peak
        re = data[:, 1]
        im = data[:, 2]

        if m == 0:
            ax.plot(t, re, color=color, lw=1.5, label=r"$\mathfrak{Re}$")
        else:
            amp = np.sqrt(re**2 + im**2)
            ax.plot(t, re,  color=color,  lw=1.5, label=r"$\mathfrak{Re}$")
            ax.plot(t, im,  color=color,  lw=1.5, label=r"$\mathfrak{Im}$",       ls="--")
            ax.plot(t, amp, color="black", lw=1.5, label=r"$|h_{\ell m}|$")

        ax.set_ylabel(rf"$\ell,m = ({l},{m})$")
        ax.legend(frameon=False, fontsize=16, loc="upper left", ncol=3)
        ax.margins(x=0)
        ax.tick_params(which="both", direction="in", top=True, right=True)

# axes[0].set_title(r"$rh_{\ell m}/M$", pad=20)
axes[-1].set_xlabel("$t~[M]$")

fig.tight_layout()
fig.align_ylabels(list(axes))
os.makedirs(OUTPUT_DIR, exist_ok=True)
out = os.path.join(OUTPUT_DIR, "athenak_inspiral_superrest.pdf")
fig.savefig(out, dpi=300, transparent=True, format="pdf", bbox_inches="tight")
print(f"Saved: {out}")
