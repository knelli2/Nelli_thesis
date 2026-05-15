import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import os

# ── Plot style (matches papers-2024-spectre-first-bbh) ───────────────────────
plt.style.use("tableau-colorblind10")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.size'] = 20
COLOR_CYCLE = ['#0F2080', '#F5793A', '#85C0F9', '#A95AA1']

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "images", "defense")

# ── Layout parameters (all in data / "figure unit" coordinates) ──────────────
CELL_SIZE  = 1.0   # side length of each square cell
CELL_GAP   = 0.15  # gap between cells in the 2x2 grid
MID_GAP    = 1.8   # white space between cell grid and bar column
BAR_WIDTH  = 1.8   # width of the AHF thread column
BAR_HEIGHT = None  # None → match total grid height / 4
CELL_LW    = 1.5   # outline linewidth for cells and bars
FILL_ALPHA = 0.25  # transparency for cell and bar fill colors
MARGIN     = 0.4   # blank border around the whole diagram

# ── Drawing function ──────────────────────────────────────────────────────────
def draw_frame(
    ax,
    cell_texts,                  # list[str] of length 4, row-major top-to-bottom, left-to-right
    bar_texts,                   # list[str] of length 4, top to bottom
    cell_colors=COLOR_CYCLE,     # list of matplotlib colors for cell fills
    bar_colors=COLOR_CYCLE,      # list of matplotlib colors for bars
    fill_alpha=FILL_ALPHA,    # transparency applied to both cell and bar fill colors
    cell_size=CELL_SIZE,
    cell_gap=CELL_GAP,
    mid_gap=MID_GAP,
    bar_width=BAR_WIDTH,
    bar_height=BAR_HEIGHT,
    cell_lw=CELL_LW,
    margin=MARGIN,
    mid_blob=None,            # text to place in a blob in the middle gap; None = no blob
    mid_blob_w=0.9,           # base width of the blob
    mid_blob_h=0.5,           # base height of the blob
    mid_blob_color='lightgrey',
    mid_blob_n_waves=4,       # frequency of the wavy perturbation
    mid_blob_amplitude=0.1,  # relative amplitude of the waviness
):
    grid_h = 2 * cell_size + cell_gap
    bh = grid_h / 4 if bar_height is None else bar_height
    if cell_colors is None:
        cell_colors = bar_colors

    # ── 2×2 cell grid ─────────────────────────────────────────────────────────
    # index order: [top-left, top-right, bottom-left, bottom-right]
    cell_positions = [(0, 1), (1, 1), (0, 0), (1, 0)]  # (col, row), row 0 = bottom
    for text, color, (col, row) in zip(cell_texts, cell_colors, cell_positions):
        x = col * (cell_size + cell_gap)
        y = row * (cell_size + cell_gap)
        ax.add_patch(mpatches.Rectangle(
            (x, y), cell_size, cell_size,
            linewidth=cell_lw, edgecolor='black', facecolor=mcolors.to_rgba(color, fill_alpha), zorder=2,
        ))
        if text:
            ax.text(
                x + cell_size / 2, y + cell_size / 2, text,
                ha='center', va='center', fontsize=mpl.rcParams['font.size'], zorder=3,
            )

    # ── Mid-gap blob ──────────────────────────────────────────────────────────
    if mid_blob is not None:
        bx = 2 * cell_size + cell_gap + mid_gap / 2
        by = grid_h / 2
        theta = np.linspace(0, 2 * np.pi, 300)
        envelope = mid_blob_amplitude * (1 + np.cos(theta))
        perturb = 1 + envelope * (
            np.sin(mid_blob_n_waves * theta) +
            0.5 * np.sin(2 * mid_blob_n_waves * theta + 1.0)
        )
        xb = bx + (mid_blob_w / 2) * perturb * np.cos(theta)
        yb = by + (mid_blob_h / 2) * perturb * np.sin(theta)
        ax.fill(xb, yb, facecolor=mid_blob_color, edgecolor='black', linewidth=cell_lw, zorder=2)
        ax.text(
            bx, by, mid_blob,
            ha='center', va='center', fontsize=mpl.rcParams['font.size'] * 0.8, zorder=3,
        )

    # ── AHF thread bar column ─────────────────────────────────────────────────
    x_bar = 2 * cell_size + cell_gap + mid_gap
    for i, (text, color) in enumerate(zip(bar_texts, bar_colors)):
        y = grid_h - (i + 1) * bh  # top to bottom
        ax.add_patch(mpatches.Rectangle(
            (x_bar, y), bar_width, bh,
            linewidth=cell_lw, edgecolor='black', facecolor=mcolors.to_rgba(color, fill_alpha), zorder=2,
        ))
        if text:
            ax.text(
                x_bar + bar_width / 2, y + bh / 2, text,
                ha='center', va='center', fontsize=mpl.rcParams['font.size'], zorder=3,
            )

    # ── Axis limits ───────────────────────────────────────────────────────────
    total_w = x_bar + bar_width
    total_h = grid_h
    ax.set_xlim(-margin, total_w + margin)
    ax.set_ylim(-margin, total_h + margin)
    ax.set_aspect('equal')
    ax.axis('off')


OUTPUT_PREFIX = "async_ahf"

# ── Frames ────────────────────────────────────────────────────────────────────
# Each entry defines one output image; files are named {OUTPUT_PREFIX}_{i}.png
FRAMES = [
    {
        "cell_texts": [r"$t_0$", r"$t_0$", r"$t_0$", r"$t_0$"],
        "bar_texts":  ["", "", "", ""],
    },
    {
        "cell_texts": [r"$t_1$", r"$t_1$", r"$t_1$", r"$t_1$"],
        "bar_texts":  [r"$t_0$", r"$t_0$", r"$t_0$", r"$t_0$"],
    },
    {
        "cell_texts": [r"$t_1$", r"$t_1$", r"$t_1$", r"$t_1$"],
        "bar_texts":  ["", "", "", ""],
    },
    {
        "cell_texts": [r"$t_1$", r"$t_1$", r"$t_1$", r"$t_0$"],
        "bar_texts":  [r"$t_0$", r"$t_0$", r"$t_0$", ""],
    },
    {
        "cell_texts": [r"$t_1$", r"$t_1$", r"$t_1$", r"$t_1$"],
        "bar_texts":  ["", "", "", r"$t_0$"],
    },
    {
        "cell_texts": [r"$t_1$", r"$t_1$", r"$t_1$", r"$t_1$"],
        "bar_texts":  ["", "", "", ""],
    },
    {
        "cell_texts": [r"$t_2$", r"$t_2$", r"$t_2$", r"$t_2$"],
        "bar_texts":  [r"$t_0,t_1$", r"$t_0$", r"$t_1$", ""],
        "mid_blob":   r"$t_0,t_1$",
    },
    {
        "cell_texts": [r"$t_2$", r"$t_2$", r"$t_2$", r"$t_2$"],
        "bar_texts":  [r"$t_1$", r"$t_1$", r"$t_1$", r"$t_1$"],
        "mid_blob":   r"$t_0$",
    },
    {
        "cell_texts": [r"$t_2$", r"$t_2$", r"$t_2$", r"$t_2$"],
        "bar_texts":  [r"$t_0,t_1$", r"$t_0,t_1$", r"$t_0,t_1$", r"$t_0,t_1$"],
    },
    {
        "cell_texts": [r"$t_2$", r"$t_2$", r"$t_2$", r"$t_2$"],
        "bar_texts":  [r"$t_1$", r"$t_1$", r"$t_1$", r"$t_1$",],
    },
]

# ── Render ────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i, frame in enumerate(FRAMES):
    grid_h = 2 * CELL_SIZE + CELL_GAP
    bh = grid_h / 4 if BAR_HEIGHT is None else BAR_HEIGHT
    total_w = 2 * CELL_SIZE + CELL_GAP + MID_GAP + BAR_WIDTH + 2 * MARGIN
    total_h = grid_h + 2 * MARGIN

    fig, ax = plt.subplots(figsize=(total_w, total_h))
    draw_frame(
        ax,
        cell_texts=frame["cell_texts"],
        bar_texts=frame["bar_texts"],
        mid_blob=frame.get("mid_blob"),
    )
    out_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{i}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)
