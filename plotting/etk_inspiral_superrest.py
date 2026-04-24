"""
Plot ETK CCE waveforms mapped to the superrest frame.
Edit CASES_TO_PLOT to select which cases to run.
"""

import os
import pickle

import numpy as np
import scri
import matplotlib as mpl
import matplotlib.pyplot as plt

from cce_common import (
    COLOR_CYCLE,
    interpolate_and_shift_cached as _interp_shift,
    map_to_superrest_cached as _map_sr,
)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.style.use("tableau-colorblind10")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.size'] = 24

# ── Choose which cases to plot ────────────────────────────────────────────────
# CASES_TO_PLOT = ["q1_aligned"]
CASES_TO_PLOT = ["q1_nonspinning"]
WT_RADIUS = 200

# ── Case → h5 filename within thesis_data/etk/<case>/ ────────────────────────
CASE_FILE = {
    "q1_nonspinning": f"CharacteristicExtractReduction_D12-mchlachlan_R{WT_RADIUS}.h5",
    "q1_aligned":     f"CharacteristicExtractReduction_D12-mchlachlan_a0.4_R{WT_RADIUS}.h5",
    "q4":             f"CharacteristicExtractReduction_D9-mchlachlan_q4_R{WT_RADIUS}.h5",
    "q1_eccentric":   f"CCE_R{WT_RADIUS}.h5",
    "q1_precessing":  f"CCE_R{WT_RADIUS}.h5",
}

# ── Paths ─────────────────────────────────────────────────────────────────────
ETK_BASE = "/home/knelli/Documents/research/sims/thesis_data/etk"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_SCRIPT_DIR, "abd_cache")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "images", "etk_cce")

# ── Frame-mapping parameters ──────────────────────────────────────────────────
DELTA_T = 1.0
OFFSET_FROM_PEAK = -1500.0 # nonspinning
# OFFSET_FROM_PEAK = -2000.0 # aligned
PADDING_TIME = 200.0

# ── Modes to plot ─────────────────────────────────────────────────────────────
MODES = [(2, 2), (2, 0), (3, 2)]


def load_abd_cached(case, h5_path):
    cache_path = os.path.join(CACHE_DIR, f"etk_{case}_raw.pkl")
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(cache_path):
        print(f"  {case}: loading raw abd from cache...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print(f"  {case}: reading from h5 ({h5_path})...")
    abd = scri.create_abd_from_h5(file_format="SpECTRECCE_v1", file_name=h5_path)
    with open(cache_path, "wb") as f:
        pickle.dump(abd, f)
    return abd


def interpolate_and_shift_cached(case, abd_raw, delta_t):
    cache_path = os.path.join(CACHE_DIR, f"etk_{case}_interp_dt{delta_t}.pkl")
    return _interp_shift(case, abd_raw, delta_t, cache_path)


def map_to_superrest_cached(case, abd, t_0, padding_time):
    cache_path = os.path.join(
        CACHE_DIR, f"etk_{case}_sr_t0{int(t_0)}_pad{int(padding_time)}.pkl"
    )
    return _map_sr(case, abd, t_0, padding_time, cache_path)


def plot_com_debug(case, abd_sr):
    com = abd_sr.bondi_CoM_charge()   # shape (N, 3): [Gx, Gy, Gz]
    t = abd_sr.t
    labels = [r"$G^x$", r"$G^y$", r"$G^z$"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (label, color) in enumerate(zip(labels, COLOR_CYCLE)):
        ax.plot(t, com[:, i], color=color, lw=1.5, label=label)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("$t~[M]$")
    ax.set_ylabel(r"$G^i~[M^2]$")
    ax.legend(frameon=False, fontsize=16)
    ax.margins(x=0)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"etk_{case}_com_charge_debug.pdf")
    fig.savefig(out, dpi=300, transparent=True, format="pdf", bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_case(case, abd_sr):
    fig, axes = plt.subplots(len(MODES), 1, figsize=(12, 14), sharex=True)

    for ax, color, (l, m) in zip(axes, COLOR_CYCLE, MODES):
        idx = abd_sr.h.index(l, m)
        t = abd_sr.t
        h = abd_sr.h.data[:, idx]
        re, im = h.real, h.imag

        if m == 0:
            ax.plot(t, re, color=color, lw=1.5, label=r"$\mathfrak{Re}$")
        else:
            amp = np.abs(h)
            ax.plot(t, re,  color=color,  lw=1.5, label=r"$\mathfrak{Re}$")
            ax.plot(t, im,  color=color,  lw=1.5, label=r"$\mathfrak{Im}$", ls="--")
            ax.plot(t, amp, color="black", lw=1.5, label=r"$|h_{\ell m}|$")

        # ax.axvline(abd_sr.t[0] + WT_RADIUS, color="gray", lw=1.0, ls="--")
        ax.set_ylabel(rf"$\ell,m = ({l},{m})$")
        ax.legend(frameon=False, fontsize=16, loc="upper left", ncol=3)
        ax.margins(x=0)
        ax.tick_params(which="both", direction="in", top=True, right=True)

    axes[-1].set_xlabel("$t~[M]$")
    fig.tight_layout()
    fig.align_ylabels(list(axes))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"etk_{case}_inspiral_superrest.pdf")
    fig.savefig(out, dpi=300, transparent=True, format="pdf", bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


for case in CASES_TO_PLOT:
    print(f"=== Processing case: {case} ===")
    h5_path = os.path.join(ETK_BASE, case, CASE_FILE[case])
    abd_raw = load_abd_cached(case, h5_path)
    abd = interpolate_and_shift_cached(case, abd_raw, DELTA_T)
    abd_sr = map_to_superrest_cached(case, abd, OFFSET_FROM_PEAK, PADDING_TIME)
    plot_case(case, abd_sr)
    # plot_com_debug(case, abd)
