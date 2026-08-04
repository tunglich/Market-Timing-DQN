"""Render docs/pipeline.png — Market-Timing-DQN 5-stage pipeline figure.

Horizontal 5-box block diagram with black borders, white fill, serif font.
Each box: bold "Stage N" title, horizontal divider, 2-line description,
italic reference annotation at the bottom.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False

OUT_PATH = Path(__file__).parent / "pipeline.png"

BG = "#ffffff"
LINE = "#000000"
FILL = "#ffffff"
TEXT = "#000000"


def _box(ax, cx, cy, w, h, title, body, ref, *,
         title_size=11.0, body_size=9.0, ref_size=8.5):
    x = cx - w / 2
    y = cy - h / 2
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.1,
        edgecolor=LINE,
        facecolor=FILL,
    )
    ax.add_patch(patch)

    # Title
    ax.text(cx, cy + h * 0.34, title, ha="center", va="center",
            color=TEXT, fontsize=title_size, fontweight="bold")
    # Divider line just under the title
    ax.plot([x + w * 0.06, x + w * 0.94],
            [cy + h * 0.18, cy + h * 0.18],
            color=LINE, linewidth=0.7)
    # Body (up to 2 lines)
    ax.text(cx, cy - h * 0.02, body, ha="center", va="center",
            color=TEXT, fontsize=body_size, linespacing=1.35)
    # Italic reference
    ax.text(cx, cy - h * 0.36, ref, ha="center", va="center",
            color=TEXT, fontsize=ref_size, fontstyle="italic")


def _arrow(ax, x1, y1, x2, y2, *, lw=1.2):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        color=LINE,
        linewidth=lw,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(a)


def main() -> None:
    fig, ax = plt.subplots(figsize=(14.0, 3.2), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box_w = 8.0
    box_h = 4.0
    cy = 5.0
    centers = [4.5, 13.5, 22.5, 31.5, 40.5, 49.5][:5]  # not used; explicit below

    xs = [4.5, 13.5, 22.5, 31.5, 40.5]

    _box(
        ax, xs[0], cy, box_w, box_h,
        "Stage 1",
        "Make Features\n(DES accuracy grid)",
        "(Eq. 2, Eq. 5)",
    )
    _box(
        ax, xs[1], cy, box_w, box_h,
        "Stage 2",
        "Walk-Forward CV\n(5 contiguous folds)",
        "(\u00a74.1)",
    )
    _box(
        ax, xs[2], cy, box_w, box_h,
        "Stage 3",
        "DQN Training\nPER + n-step Double DQN",
        "(\u00a74.2)",
    )
    _box(
        ax, xs[3], cy, box_w, box_h,
        "Stage 4",
        "Backtest\nCost-Adjusted",
        "(Table 4)",
    )
    _box(
        ax, xs[4], cy, box_w, box_h,
        "Stage 5",
        "Portfolio\nEqual / Cap / Price",
        "(Figure 7)",
    )

    for i in range(4):
        x1 = xs[i] + box_w / 2
        x2 = xs[i + 1] - box_w / 2
        _arrow(ax, x1, cy, x2, cy)

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[OK] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
