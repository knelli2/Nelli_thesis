import os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scri

# ── Plot style (matches papers-2024-spectre-first-bbh) ───────────────────────
plt.style.use("tableau-colorblind10")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.size'] = 24

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = "/home/knelli/Documents/research/sims/lovelace_nelli_bbh_cce"

LEVELS = {
    "Lev0": "Lev0_Joined/CharacteristicExtractReduction.h5",
    "Lev1": "Lev1_Joined/CharacteristicExtractReduction.h5",
    "Lev2": "RERUN111824_Lev2_Joined/CharacteristicExtractReduction.h5",
}

DELTA_T = 1.0  # [M] downsampling interval — increase to speed up

# Method 1: t_0 auto-computed per level as (time of peak |h_22|) + M1_T_OFFSET
M1_T_OFFSET = 400.0   # [M] offset from peak strain to superrest mapping time
M1_PADDING_TIME = 100.0

# Method 2: Lev2 mapped to superrest, Lev0/Lev1 mapped to Lev2 frame
M2_T_0 = 2000.0
M2_PADDING_TIME = 200.0

PLOT_MODES = [(2, 2), (2, 0), (2, 1)]  # (ell, m) modes to compare

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "images", "cross_code_cce")
CACHE_DIR = os.path.join(_SCRIPT_DIR, "abd_cache")

# ── Load & downsample ─────────────────────────────────────────────────────────
def load_abd(rel_path):
    abd = scri.create_abd_from_h5(
        file_format="SpECTRECCE_v1",
        file_name=f"{BASE_DIR}/{rel_path}",
    )
    new_t = np.arange(abd.t[0], abd.t[-1], DELTA_T)
    return abd.interpolate(new_t)


print("Loading waveform data...")
abds_raw = {label: load_abd(path) for label, path in LEVELS.items()}
print("Done loading.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def find_peak_time(abd):
    """Return the time of peak |h_22| amplitude."""
    idx22 = abd.h.index(2, 2)
    amp = np.abs(abd.h.data[:, idx22])
    return abd.h.t[np.argmax(amp)]


def get_h_mode(abd, mode):
    idx = abd.h.index(mode[0], mode[1])
    return abd.h.t, abd.h.data[:, idx]


def amplitude(h):
    return np.abs(h)


def phase(h):
    return np.arctan(h.imag / h.real)


def compute_diffs(abds, mode, lo_label, hi_label):
    """Fractional amplitude diff and phase diff of lo vs hi, on hi's time grid."""
    t_hi, h_hi = get_h_mode(abds[hi_label], mode)
    t_lo, h_lo = get_h_mode(abds[lo_label], mode)

    amp_hi = amplitude(h_hi)
    amp_lo_interp = np.interp(t_hi, t_lo, amplitude(h_lo))
    frac_amp_diff = np.abs(amp_lo_interp - amp_hi) / amp_hi

    phase_hi = phase(h_hi)
    phase_lo_interp = np.interp(t_hi, t_lo, phase(h_lo))
    phase_diff = phase_lo_interp - phase_hi

    return t_hi, frac_amp_diff, phase_diff


# ── Pickle cache helpers ──────────────────────────────────────────────────────
def _load_cache(cache_path, t_0, padding_time):
    """Return cached abd if t_0 and padding_time match, else None."""
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    if cache["t_0"] == t_0 and cache["padding_time"] == padding_time:
        return cache["abd"]
    return None


def _save_cache(cache_path, abd, t_0, padding_time):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"abd": abd, "t_0": t_0, "padding_time": padding_time}, f)


def map_to_superrest_cached(label, abd_raw, t_0, padding_time, cache_path):
    cached = _load_cache(cache_path, t_0, padding_time)
    if cached is not None:
        print(f"  {label}: using cached superrest mapping (t_0={t_0:.1f}, padding={padding_time:.1f})")
        return cached
    print(f"  {label}: mapping to superrest frame (t_0={t_0:.1f}, padding={padding_time:.1f})...")
    abd_sr, _, _ = abd_raw.map_to_superrest_frame(t_0=t_0, padding_time=padding_time)
    _save_cache(cache_path, abd_sr, t_0, padding_time)
    return abd_sr


def map_to_abd_frame_cached(label, abd_raw, target_abd, t_0, padding_time, cache_path):
    cached = _load_cache(cache_path, t_0, padding_time)
    if cached is not None:
        print(f"  {label}: using cached abd frame mapping (t_0={t_0:.1f}, padding={padding_time:.1f})")
        return cached
    print(f"  {label}: mapping to Lev2 frame (t_0={t_0:.1f}, padding={padding_time:.1f})...")
    abd_mapped, _, _ = abd_raw.map_to_abd_frame(
        target_abd, t_0=t_0, padding_time=padding_time, fix_time_phase_freedom=False
    )
    _save_cache(cache_path, abd_mapped, t_0, padding_time)
    return abd_mapped


# ── Method 1: each level individually to superrest ────────────────────────────
print("Method 1: mapping each level to superrest frame...")
abds_m1 = {}
for label, abd in abds_raw.items():
    t_peak = find_peak_time(abd)
    t_0 = t_peak + M1_T_OFFSET
    cache_path = os.path.join(CACHE_DIR, f"m1_{label.lower()}.pkl")
    abds_m1[label] = map_to_superrest_cached(label, abd, t_0, M1_PADDING_TIME, cache_path)
print("Method 1 done.")

# ── Method 2: Lev2 to superrest, Lev0/Lev1 mapped to Lev2 frame ──────────────
print("Method 2: mapping Lev2 to superrest frame...")
cache_path_lev2_sr = os.path.join(CACHE_DIR, "m2_lev2_sr.pkl")
abd_lev2_sr = map_to_superrest_cached(
    "Lev2", abds_raw["Lev2"], M2_T_0, M2_PADDING_TIME, cache_path_lev2_sr
)

abds_m2 = {"Lev2": abd_lev2_sr}
for label in ["Lev0", "Lev1"]:
    cache_path = os.path.join(CACHE_DIR, f"m2_{label.lower()}_mapped.pkl")
    abds_m2[label] = map_to_abd_frame_cached(
        label, abds_raw[label], abd_lev2_sr, M2_T_0, M2_PADDING_TIME, cache_path
    )
print("Method 2 done.")

# ── Plotting ──────────────────────────────────────────────────────────────────
PAIRS = [("Lev0", "Lev1"), ("Lev1", "Lev2")]  # (lo, hi)
COLOR_CYCLE = ['#0F2080', '#F5793A', '#85C0F9', '#A95AA1']
COLORS = {"Lev0-Lev1": COLOR_CYCLE[0], "Lev1-Lev2": COLOR_CYCLE[1]}


def make_comparison_figure(abds, title, filename):
    n_modes = len(PLOT_MODES)
    fig, axes = plt.subplots(
        n_modes, 2, figsize=(16, 5 * n_modes), sharex=False, squeeze=False
    )
    fig.suptitle(title)

    for row, mode in enumerate(PLOT_MODES):
        ell, m = mode
        ax_amp = axes[row, 0]
        ax_phase = axes[row, 1]

        for lo_label, hi_label in PAIRS:
            pair_key = f"{lo_label}-{hi_label}"
            t, frac_amp_diff, phase_diff = compute_diffs(abds, mode, lo_label, hi_label)
            ax_amp.plot(t, frac_amp_diff, label=pair_key, color=COLORS[pair_key])
            ax_phase.plot(t, phase_diff, label=pair_key, color=COLORS[pair_key])

        ax_amp.set_ylabel(rf"$|\Delta h_{{{ell}{m}}}|/|h_{{{ell}{m}}}|$")
        ax_phase.set_ylabel(rf"$\Delta\phi_{{{ell}{m}}}$")
        ax_amp.set_xlabel("$t~[M]$")
        ax_phase.set_xlabel("$t~[M]$")

        ax_amp.set_yscale("log")
        ax_phase.set_yscale("log")

        ax_amp.legend(frameon=False)

    axes[0, 0].set_title(r"Fractional amplitude difference")
    axes[0, 1].set_title(r"Phase difference")

    for axis in axes.flat:
        axis.label_outer()

    plt.subplots_adjust(hspace=0.0, wspace=0.15)
    out_path = f"{OUTPUT_DIR}/{filename}"
    fig.savefig(out_path, dpi=300, transparent=True, format='pdf', bbox_inches='tight')
    print(f"Saved: {out_path}")


os.makedirs(OUTPUT_DIR, exist_ok=True)

make_comparison_figure(abds_m1, "Remnant superrest frame", "method1_superrest_comparison.pdf")
make_comparison_figure(abds_m2, "Lev2 inspiral superrest frame", "method2_abd_frame_comparison.pdf")
