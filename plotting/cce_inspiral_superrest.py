"""Method 2: Lev2 mapped to superrest frame, Lev0/Lev1 mapped to Lev2 BMS frame."""

import os

import numpy as np

from cce_common import (
    LEVELS, CACHE_DIR, OUTPUT_DIR,
    find_peak_time, load_abd,
    map_to_superrest_cached, map_to_abd_frame_cached,
    make_comparison_figure,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DELTA_T = 1.0       # [M] downsampling interval
T_0 = -2000.0       # [M] in peak-recentered time: mapping time during inspiral
PADDING_TIME = 200.0

# ── Load raw ABDs ─────────────────────────────────────────────────────────────
print("Loading waveform data...")
abds_raw = {label: load_abd(label, path) for label, path in LEVELS.items()}
print("Done loading.")

# ── Recenter time at peak |h_22| and downsample ───────────────────────────────
abds = {}
for label, abd in abds_raw.items():
    t_peak = find_peak_time(abd)
    abd.t = abd.t - t_peak
    new_t = np.arange(abd.t[0], abd.t[-1], DELTA_T)
    print(f"{label}: start={new_t[0]:.1f}, end={new_t[-1]:.1f}, original peak={t_peak:.1f}")
    abds[label] = abd.interpolate(new_t)

# ── Map Lev2 to superrest, then Lev0/Lev1 to Lev2 frame ─────────────────────
print("Mapping Lev2 to superrest frame...")
cache_path_lev2_sr = os.path.join(CACHE_DIR, "m2_lev2_sr.pkl")
abd_lev2_sr = map_to_superrest_cached(
    "Lev2", abds["Lev2"], T_0, PADDING_TIME, cache_path_lev2_sr
)

abds_sr = {"Lev2": abd_lev2_sr}
for label in ["Lev0", "Lev1"]:
    cache_path = os.path.join(CACHE_DIR, f"m2_{label.lower()}_mapped.pkl")
    abds_sr[label] = map_to_abd_frame_cached(
        label, abds[label], abd_lev2_sr, T_0, PADDING_TIME, cache_path
    )
print("Done.")

# ── Plot ──────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
make_comparison_figure(
    abds_sr,
    title="Lev2 inspiral superrest frame",
    filename="method2_abd_frame_comparison.pdf",
    debug_amp_col=True,
)
