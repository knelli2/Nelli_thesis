import os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

PLOT_MODES = [(2, 2), (2, 0), (3, 2), (4, 4)]  # (ell, m) modes to compare

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "images", "cross_code_cce")
CACHE_DIR = os.path.join(_SCRIPT_DIR, "abd_cache")

# ── Plotting constants ────────────────────────────────────────────────────────
PAIRS = [("Lev0", "Lev1"), ("Lev1", "Lev2")]  # (lo, hi)
COLOR_CYCLE = ['#0F2080', '#F5793A', '#85C0F9', '#A95AA1']
COLORS = {"Lev0-Lev1": COLOR_CYCLE[0], "Lev1-Lev2": COLOR_CYCLE[1]}
LEVEL_COLORS = {"Lev0": COLOR_CYCLE[0], "Lev1": COLOR_CYCLE[1], "Lev2": COLOR_CYCLE[2]}


# ── Waveform helpers ──────────────────────────────────────────────────────────

def find_peak_time(abd):
    """Return the time of peak |h_22| amplitude."""
    idx22 = abd.h.index(2, 2)
    amp = np.abs(abd.h.data[:, idx22])
    return abd.t[np.argmax(amp)]


def load_abd(label, rel_path):
    """Load raw ABD from h5 or pickle cache. No time recentering or interpolation."""
    cache_path = os.path.join(CACHE_DIR, f"raw_{label.lower()}.pkl")
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(cache_path):
        print(f"  {label}: loading raw abd from cache...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print(f"  {label}: reading from h5...")
    abd = scri.create_abd_from_h5(
        file_format="SpECTRECCE_v1",
        file_name=f"{BASE_DIR}/{rel_path}",
    )
    with open(cache_path, "wb") as f:
        pickle.dump(abd, f)
    return abd


def get_h_mode(abd, mode):
    idx = abd.h.index(mode[0], mode[1])
    return abd.t, abd.h.data[:, idx]


def amplitude(h, m_is_zero: bool):
    return h.real if m_is_zero else np.abs(h)


def phase(h):
    return np.unwrap(np.angle(h))
    # return np.arctan2(h.imag , h.real)


def compute_diffs(abds, mode, lo_label, hi_label, ref_time):
    """Fractional amplitude diff and phase diff of lo vs hi. ABDs must share a common time grid.

    Phase of each level is zeroed at ref_time before computing the difference.
    """
    t, h_hi = get_h_mode(abds[hi_label], mode)
    _, h_lo = get_h_mode(abds[lo_label], mode)

    m_is_zero = mode[1] == 0
    amp_hi = amplitude(h_hi, m_is_zero)
    frac_amp_diff = amplitude(h_lo, m_is_zero) - amp_hi
    frac_amp_diff = np.abs(frac_amp_diff)
    frac_amp_diff /= np.abs(amp_hi)

    ref_idx = np.argmin(np.abs(t - ref_time))
    phase_hi = phase(h_hi)
    phase_lo = phase(h_lo)
    phase_hi = phase_hi - phase_hi[ref_idx]
    phase_lo = phase_lo - phase_lo[ref_idx]
    phase_diff = np.abs(phase_lo - phase_hi)

    return t, frac_amp_diff, phase_diff


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


def interpolate_and_shift_cached(label, abd_raw, delta_t, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if cache["delta_t"] == delta_t:
            print(f"  {label}: using cached interpolated abd (delta_t={delta_t})")
            return cache["abd"]
    print(f"  {label}: interpolating and shifting (delta_t={delta_t})...")
    new_t = np.arange(abd_raw.t[0], abd_raw.t[-1], delta_t)
    abd = abd_raw.interpolate(new_t).t_shift_peak_to_zero()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"abd": abd, "delta_t": delta_t}, f)
    return abd


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


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_comparison_figure(abds, title, filename, ref_time, debug_amp_col=False):
    n_modes = len(PLOT_MODES)
    n_cols = 3 if debug_amp_col else 2
    fig_width = 24 if debug_amp_col else 16
    col_offset = 1 if debug_amp_col else 0

    fig = plt.figure(figsize=(fig_width, 5 * n_modes))
    fig.suptitle(title)
    gs = gridspec.GridSpec(n_modes, n_cols, figure=fig, hspace=0.0, wspace=0.2)

    first_amp_ax = first_phase_ax = None
    sci_debug_axes = []  # debug axes that use scientific notation
    phase_axes = []  # non-zero m phase axes for exact tick setting
    for row, mode in enumerate(PLOT_MODES):
        ell, m = mode
        bottom = row == n_modes - 1

        # ── Debug column: amplitude of each level for this mode ───────────────
        if debug_amp_col:
            ax_debug = fig.add_subplot(gs[row, 0])
            for label in sorted(abds.keys()):
                t, h = get_h_mode(abds[label], mode)
                ax_debug.plot(t, amplitude(h, mode[1] ==0), label=label, color=LEVEL_COLORS[label])
            ax_debug.set_yscale("linear")
            ax_debug.margins(x=0)
            ax_debug.set_ylabel(rf"$\ell,m = ({ell},{m})$")
            if row == 0:
                ax_debug.legend(frameon=False)
                ax_debug.set_title(r"$|rh_{\ell m}|$", pad=20)
            else:
                ax_debug.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                sci_debug_axes.append(ax_debug)
            if bottom:
                ax_debug.set_xlabel("$t~[M]$")
            else:
                ax_debug.tick_params(labelbottom=False)
        # ── End debug column ──────────────────────────────────────────────────

        ax_amp = fig.add_subplot(gs[row, col_offset])
        ax_phase = fig.add_subplot(gs[row, col_offset + 1])

        for lo_label, hi_label in PAIRS:
            pair_key = f"{lo_label}-{hi_label}"
            t, frac_amp_diff, phase_diff = compute_diffs(abds, mode, lo_label, hi_label, ref_time)
            ax_amp.plot(t, frac_amp_diff, label=pair_key, color=COLORS[pair_key])
            if m != 0:
                ax_phase.plot(t, phase_diff, label=pair_key, color=COLORS[pair_key])

        if not debug_amp_col:
            ax_amp.set_ylabel(rf"$\ell,m = ({ell},{m})$")
        ax_amp.set_yscale("log")
        ax_amp.margins(x=0)
        if row == 0:
            ax_amp.legend(frameon=False)

        if m == 0:
            ax_phase.text(
                0.5, 0.5, "This mode is real",
                transform=ax_phase.transAxes,
                ha="center", va="center",
            )
            ax_phase.set_yticks([])
        else:
            ax_phase.set_yscale("log")
            ax_phase.margins(x=0)
            phase_axes.append(ax_phase)
            if row == 0:
                ax_phase.legend(frameon=False)

        if bottom:
            ax_amp.set_xlabel("$t~[M]$")
            ax_phase.set_xlabel("$t~[M]$")
        else:
            ax_amp.tick_params(labelbottom=False)
            ax_phase.tick_params(labelbottom=False)

        if first_amp_ax is None:
            first_amp_ax = ax_amp
            first_phase_ax = ax_phase

    first_amp_ax.set_title(r"$|\Delta h_{\ell m}| / |h_{\ell m}|$", pad=20)
    first_phase_ax.set_title(r"$|\Delta\phi_{\ell m}|$", pad=20)

    # Set exactly 4 y-ticks on each phase column (limits must be finalized first)
    fig.canvas.draw()
    for ax in phase_axes:
        ymin, ymax = ax.get_ylim()
        ticks = [1e-5, 1e-3, 1e-1, 1e1]
        print(ticks)
        ax.set_yticks(ticks)

    # Move sci-notation exponent inside each debug panel
    if sci_debug_axes:
        for ax in sci_debug_axes:
            ot = ax.yaxis.get_offset_text()
            exponent_str = ot.get_text()
            ot.set_visible(False)
            if exponent_str:
                ax.text(0.02, 0.97, exponent_str, transform=ax.transAxes,
                        ha='left', va='top', fontsize=mpl.rcParams['font.size'] * 0.7)

    out_path = f"{OUTPUT_DIR}/{filename}"
    fig.savefig(out_path, dpi=300, transparent=True, format='pdf', bbox_inches='tight')
    print(f"Saved: {out_path}")
