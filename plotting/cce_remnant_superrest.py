"""Method 1: each level individually mapped to the superrest frame post-ringdown."""

import os

import numpy as np

from cce_common import (
    LEVELS, CACHE_DIR, OUTPUT_DIR,
    find_peak_time, load_abd,
    map_to_superrest_cached,
    make_comparison_figure,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DELTA_T = 1.0                     # [M] downsampling interval
OFFSET_AFTER_PEAK = 200.0         # [M] in peak-recentered time: mapping time after peak
PADDING_TIME = 100.0

# ── Load raw ABDs ─────────────────────────────────────────────────────────────
print("Loading waveform data...")
abds_raw = {label: load_abd(label, path) for label, path in LEVELS.items()}
print("Done loading.")

# ── Recenter time at peak |h_22| and downsample ───────────────────────────────
abds = {}
for label, abd in abds_raw.items():
    new_t = np.arange(abd.t[0], abd.t[-1], DELTA_T)
    abds[label] = abd.interpolate(new_t)

# ── Map each level to superrest frame ─────────────────────────────────────────
print("Mapping each level to superrest frame...")
abds_sr = {}
for label, abd in abds.items():
    cache_path = os.path.join(CACHE_DIR, f"m1_{label.lower()}.pkl")
    t_peak = find_peak_time(abd)
    t_0 = t_peak + OFFSET_AFTER_PEAK
    abds_sr[label] = map_to_superrest_cached(label, abd, t_0, PADDING_TIME, cache_path)
print("Done.")

# ── Interpolate then shift so all ABDs share an identical time array ──────────
t_peaks = {label: find_peak_time(abd) for label, abd in abds_sr.items()}

t_min = max(abd.t[0] - t_peaks[label] for label, abd in abds_sr.items())
t_max = min(abd.t[-1] - t_peaks[label] for label, abd in abds_sr.items())
common_t = np.arange(t_min, t_max, DELTA_T)
print(f"Common post-shift grid: t=[{t_min:.1f}, {t_max:.1f}], N={len(common_t)}")

abds_sr_intrp = {}
for label, abd in abds_sr.items():
    abd_interp = abd.interpolate(common_t + t_peaks[label])
    abd_interp.t = abd_interp.t - t_peaks[label]
    abds_sr_intrp[label] = abd_interp

# ── Plot ──────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
make_comparison_figure(
    abds_sr_intrp,
    title="Remnant superrest frame",
    filename="spectre_remnant_superrest_comparison.pdf",
    debug_amp_col=True,
)
