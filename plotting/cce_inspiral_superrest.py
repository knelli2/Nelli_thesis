"""Method 2: Lev2 mapped to superrest frame, Lev0/Lev1 mapped to Lev2 BMS frame."""

import os

import numpy as np

from cce_common import (
    LEVELS, CACHE_DIR, OUTPUT_DIR, COLOR_CYCLE,
    load_abd, interpolate_and_shift_cached,
    map_to_superrest_cached, map_to_abd_frame_cached,
    make_comparison_figure,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DELTA_T = 1.0                   # [M] downsampling interval
OFFSET_FROM_PEAK = -3000.0       # [M] in peak-recentered time: mapping time during inspiral
PADDING_TIME = 200.0
PHASE_REF_TIME = -4000.0         # [M] time at which phase difference is set to zero

# ── Plotting constants ─────────────────────────────────────────────────────────
PLOT_MODES = [(2, 2), (2, 0), (3, 2), (4, 4)]  # (ell, m) modes to compare
PAIRS = [("Lev0", "Lev1"), ("Lev1", "Lev2")]   # (lo, hi)
COLORS = {"Lev0-Lev1": COLOR_CYCLE[0], "Lev1-Lev2": COLOR_CYCLE[1]}
LEVEL_COLORS = {"Lev0": COLOR_CYCLE[0], "Lev1": COLOR_CYCLE[1], "Lev2": COLOR_CYCLE[2]}

# ── Load raw ABDs ─────────────────────────────────────────────────────────────
print("Loading waveform data...")
abds_raw = {label: load_abd(label, path) for label, path in LEVELS.items()}
print("Done loading.")

# ── Recenter time at peak |h_22| and downsample ───────────────────────────────
print(f"Interpolating to dt = {DELTA_T} and time-shifting")
abds = {}
for label, abd_raw in abds_raw.items():
    cache_path = os.path.join(CACHE_DIR, f"interp_{label.lower()}_dt{DELTA_T}.pkl")
    abds[label] = interpolate_and_shift_cached(label, abd_raw, DELTA_T, cache_path)
print("Done interpolating and shifting.")

# ── Map Lev2 to superrest, then Lev0/Lev1 to Lev2 frame ─────────────────────
print("Mapping Lev2 to superrest frame...")
cache_path_lev2_sr = os.path.join(CACHE_DIR, "m2_lev2_sr.pkl")
abd_lev2_sr = map_to_superrest_cached(
    "Lev2", abds["Lev2"], OFFSET_FROM_PEAK, PADDING_TIME, cache_path_lev2_sr
)

abds_sr = {"Lev2": abd_lev2_sr}
for label in ["Lev0", "Lev1"]:
    cache_path = os.path.join(CACHE_DIR, f"m2_{label.lower()}_mapped.pkl")
    abds_sr[label] = map_to_abd_frame_cached(
        label, abds[label], abd_lev2_sr, OFFSET_FROM_PEAK, PADDING_TIME, cache_path
    )
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
    title="Inspiral superrest frame",
    filename="spectre_inspiral_superrest_comparison.pdf",
    ref_time=PHASE_REF_TIME,
    plot_modes=PLOT_MODES,
    pairs=PAIRS,
    colors=COLORS,
    level_colors=LEVEL_COLORS,
    debug_amp_col=True,
)
