"""Method 2: Lev2 mapped to superrest frame, Lev0/Lev1 mapped to Lev2 BMS frame."""

import os

import numpy as np

from cce_common import (
    CACHE_DIR, OUTPUT_DIR, COLOR_CYCLE,
    load_abd, interpolate_and_shift_cached,
    map_to_superrest_cached, map_to_abd_frame_cached,
    make_comparison_figure,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DELTA_T = 1.0                   # [M] downsampling interval
OFFSET_FROM_PEAK = -3000.0       # [M] in peak-recentered time: mapping time during inspiral
PADDING_TIME = 200.0
PHASE_REF_TIME = -4000.0         # [M] time at which phase difference is set to zero

BASE_DIR = "/home/knelli/Documents/research/sims/thesis_data/fil_spec"

CODES = [
    "FIL",
    "SpEC",
]

# ── Plotting constants ─────────────────────────────────────────────────────────
PLOT_MODES = [(2, 2), (2, 0), (3, 2), (4, 4)]  # (ell, m) modes to compare
PAIRS = [("SpEC", "FIL"),]   # (lo, hi)
COLORS = {"SpEC-FIL": COLOR_CYCLE[0],}
LEVEL_COLORS = {"SpEC": COLOR_CYCLE[0], "FIL": COLOR_CYCLE[1],}

# ── Load raw ABDs ─────────────────────────────────────────────────────────────
print("Loading waveform data...")
abds_raw = {code: load_abd(code, f"{BASE_DIR}/CharacteristicExtractReduction_{code}.h5") for code in CODES}
print("Done loading.")

# ── Recenter time at peak |h_22| and downsample ───────────────────────────────
print(f"Interpolating to dt = {DELTA_T} and time-shifting")
abds = {}
for label, abd_raw in abds_raw.items():
    cache_path = os.path.join(CACHE_DIR, f"interp_{label.lower()}_dt{DELTA_T}.pkl")
    abds[label] = interpolate_and_shift_cached(label, abd_raw, DELTA_T, cache_path)
print("Done interpolating and shifting.")

# ── Map SpEC to superrest, then FIL to SpEC frame ─────────────────────
print("Mapping SpEC to superrest frame...")
cache_path_spec_sr = os.path.join(CACHE_DIR, "spec_fil_spec_sr.pkl")
abd_spec_sr = map_to_superrest_cached(
    "SpEC", abds["SpEC"], OFFSET_FROM_PEAK, PADDING_TIME, cache_path_spec_sr
)

abds_sr = {"SpEC": abd_spec_sr}
cache_path_fil_sr = os.path.join(CACHE_DIR, "spec_fil_fil_sr.pkl")
abds_sr["FIL"] = map_to_abd_frame_cached(
    "FIL", abds["FIL"], abd_spec_sr, OFFSET_FROM_PEAK, PADDING_TIME, cache_path_fil_sr
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
    filename="spec_fil_inspiral_superrest_comparison.pdf",
    ref_time=PHASE_REF_TIME,
    plot_modes=PLOT_MODES,
    pairs=PAIRS,
    colors=COLORS,
    level_colors=LEVEL_COLORS,
    debug_amp_col=True,
)
