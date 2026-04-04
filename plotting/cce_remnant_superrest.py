"""Method 1: each level individually mapped to the superrest frame post-ringdown."""

import os

import numpy as np

from cce_common import (
    LEVELS, BASE_DIR, CACHE_DIR, OUTPUT_DIR, COLOR_CYCLE,
    load_abd, interpolate_and_shift_cached,
    map_to_superrest_cached,
    make_comparison_figure,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DELTA_T = 1.0                     # [M] downsampling interval
OFFSET_AFTER_PEAK = 200.0         # [M] in peak-recentered time: mapping time after peak
PADDING_TIME = 100.0
PHASE_REF_TIME = 0.0              # [M] time at which phase difference is set to zero

# ── Plotting constants ─────────────────────────────────────────────────────────
PLOT_MODES = [(2, 2), (2, 0), (3, 2), (4, 4)]  # (ell, m) modes to compare
PAIRS = [("Lev0", "Lev1"), ("Lev1", "Lev2")]   # (lo, hi)
COLORS = {"Lev0-Lev1": COLOR_CYCLE[0], "Lev1-Lev2": COLOR_CYCLE[1]}
LEVEL_COLORS = {"Lev0": COLOR_CYCLE[0], "Lev1": COLOR_CYCLE[1], "Lev2": COLOR_CYCLE[2]}

# ── Load raw ABDs ─────────────────────────────────────────────────────────────
print("Loading waveform data...")
abds_raw = {label: load_abd(label, f"{BASE_DIR}/{path}") for label, path in LEVELS.items()}
print("Done loading.")

# ── Recenter time at peak |h_22| and downsample ───────────────────────────────
abds = {}
for label, abd_raw in abds_raw.items():
    cache_path = os.path.join(CACHE_DIR, f"interp_{label.lower()}_dt{DELTA_T}.pkl")
    abds[label] = interpolate_and_shift_cached(label, abd_raw, DELTA_T, cache_path)

# ── Map each level to superrest frame ─────────────────────────────────────────
print("Mapping each level to superrest frame...")
abds_sr = {}
for label, abd in abds.items():
    cache_path = os.path.join(CACHE_DIR, f"m1_{label.lower()}.pkl")
    abds_sr[label] = map_to_superrest_cached(label, abd, OFFSET_AFTER_PEAK, PADDING_TIME, cache_path)
print("Done.")

# ── Interpolate all ABDs share an identical time array ──────────
t_min = max(abd.t[0] for _, abd in abds_sr.items())
t_max = min(abd.t[-1] for _, abd in abds_sr.items())
common_t = np.arange(t_min, t_max, DELTA_T)
print(f"Common post-shift grid: t=[{t_min:.1f}, {t_max:.1f}], N={len(common_t)}")

abds_sr_intrp = {}
for label, abd in abds_sr.items():
    abds_sr_intrp[label] = abd.interpolate(common_t)

# ── Plot ──────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
make_comparison_figure(
    abds_sr_intrp,
    title="Remnant superrest frame",
    filename="spectre_remnant_superrest_comparison.pdf",
    ref_time=PHASE_REF_TIME,
    plot_modes=PLOT_MODES,
    pairs=PAIRS,
    colors=COLORS,
    level_colors=LEVEL_COLORS,
    debug_amp_col=True,
)
