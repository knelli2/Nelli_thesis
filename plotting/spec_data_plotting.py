import h5py
import numpy as np
import pickle
import scri
from scri.bms_transformations import BMSTransformation
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from quaternion.calculus import indefinite_integral as integrate
from scri.asymptotic_bondi_data.map_to_superrest_frame import MT_to_WM, WM_to_MT
from spherical_functions import constant_from_ell_0_mode
from cce_common import (
    CACHE_DIR, OUTPUT_DIR,
    make_comparison_figure,
    find_peak_time,
)

# ── Plot style (matches papers-2024-spectre-first-bbh) ───────────────────────
plt.style.use("tableau-colorblind10")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.size'] = 24

# ── Configuration ─────────────────────────────────────────────────────────────
DELTA_T = 1.0
PHASE_REF_TIME = -8000.0         # [M] time at which phase difference is set to zero

BASE_DIR = "/home/knelli/Documents/research/sims/thesis_data/SXS:BBH:2696"

LEVELS = [
    "Lev2",
    "Lev3",
    "Lev4",
]

PLOT_MODES = [(2, 2), (2, 0), (2, 1), (3, 2)]  # (ell, m) modes to compare

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "images", "cross_code_cce")

# ── Plotting constants ────────────────────────────────────────────────────────
PAIRS = [("Lev2", "Lev3"), ("Lev3", "Lev4")]  # (lo, hi)
COLOR_CYCLE = ['#0F2080', '#F5793A', '#85C0F9', '#A95AA1']
COLORS = {"Lev2-Lev3": COLOR_CYCLE[0], "Lev3-Lev4": COLOR_CYCLE[1]}
LEVEL_COLORS = {"Lev2": COLOR_CYCLE[0], "Lev3": COLOR_CYCLE[1], "Lev4": COLOR_CYCLE[2]}

os.makedirs(CACHE_DIR, exist_ok=True)

abds = {}
for level in LEVELS:
    path_to_cce_simulation = f"{BASE_DIR}/{level}"
    bms = BMSTransformation()
    bms.from_file(f"{path_to_cce_simulation}/BMS.h5", group="inspiral_superrest")
    time_shift = constant_from_ell_0_mode(bms.supertranslation[0])

    cache_path = os.path.join(CACHE_DIR, f"spec_abd_prime_{level.lower()}.pkl")
    if os.path.exists(cache_path):
        print(f"  {level}: loading abd_prime from cache...")
        with open(cache_path, "rb") as f:
            abds[level] = pickle.load(f)
        continue

    path_to_cce_simulation = f"{BASE_DIR}/{level}"
    print(f"  {level}: reading from h5...")

    with h5py.File(
            f"{path_to_cce_simulation}/ExtraWaveforms.h5"
    ) as input_file:
        keys = list(input_file.keys())
        radii = [key.split("_R")[1][:4] for key in keys if "rMPsi4" in key]
        not_input_radii = [
            key.split("_R")[1][:4] for key in keys if "rhOverM" in key
        ]
        input_radius = [
            radius for radius in radii if not radius in not_input_radii
        ][0]

    abd = scri.SpEC.file_io.create_abd_from_h5(
        h=f"{path_to_cce_simulation}/Strain_CCE.h5",
        Psi4=f"{path_to_cce_simulation}/ExtraWaveforms.h5/rMPsi4_BondiCce_R{input_radius}",
        Psi3=f"{path_to_cce_simulation}/ExtraWaveforms.h5/r2Psi3_BondiCce_R{input_radius}",
        Psi2=f"{path_to_cce_simulation}/ExtraWaveforms.h5/r3Psi2OverM_BondiCce_R{input_radius}",
        Psi1=f"{path_to_cce_simulation}/ExtraWaveforms.h5/r4Psi1OverM2_BondiCce_R{input_radius}",
        Psi0=f"{path_to_cce_simulation}/ExtraWaveforms.h5/r5Psi0OverM3_BondiCce_R{input_radius}",
        file_format="RPDMB",
    )

    print(f"  {level}: applying BMS transformation...")
    abd_prime = abd.transform(
        supertranslation=bms.supertranslation,
        frame_rotation=bms.frame_rotation.components,
        boost_velocity=bms.boost_velocity
    )

    abd_peak = abd_prime.t_shift_peak_to_zero()

    with open(cache_path, "wb") as f:
        pickle.dump(abd_peak, f)
    abds[level] = abd_peak

def printTimeToSuperrest():
    news = MT_to_WM(2.0 * abds["Lev2"].sigma.bar.dot, dataType=scri.hdot)
    omega = np.linalg.norm(news.angular_velocity(), axis=1)
    phase = integrate(omega, news.t)
    t_relax = 1231.5
    t_shift = news.t[
        np.argmin(
            abs(
                phase
                - (phase[np.argmin(abs(news.t - t_relax))] + 3 * (2 * np.pi))
            )
        )
    ]

    time = abds["Lev2"].t[0] + t_relax + 0.5 * t_shift
    print(f"Superrest time = {time}")

printTimeToSuperrest()

t_min = max(abd.t[0] for _, abd in abds.items())
t_max = min(abd.t[-1] for _, abd in abds.items())
common_t = np.arange(t_min, t_max, DELTA_T)
print(f"Common post-shift grid: t=[{t_min:.1f}, {t_max:.1f}], N={len(common_t)}")

abds_intrp = {}
for label, abd in abds.items():
    abds_intrp[label] = abd.interpolate(common_t)

# ── Plot ──────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
make_comparison_figure(
    abds_intrp,
    title="Inspiral superrest frame",
    filename="spec_inspiral_superrest_comparison.pdf",
    ref_time=PHASE_REF_TIME,
    plot_modes=PLOT_MODES,
    pairs=PAIRS,
    colors=COLORS,
    level_colors=LEVEL_COLORS,
    debug_amp_col=True,
)
