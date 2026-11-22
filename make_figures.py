"""Draw figures/ from results/. Run t04_rlhf.py first, fig_loading.png uses its work files."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import HERE, RESULTS

OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({"font.family": ["Calibri", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def box(ax, x, y, w, h, text, fc="#f2f2f0", ec=MUTED, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, color=color,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, color=INK2, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12, color=color,
                                 lw=1.4, linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def fig_loop():
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4))
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.axis("off")

    ax = axes[0]
    ax.text(0, 2.85, "What the paper describes", fontsize=12, fontweight="bold", color=INK)
    box(ax, 0.1, 1.2, 1.9, 0.9, "MINIM\n(the image maker)", fc="#e6effa", ec=BLUE, bold=True)
    box(ax, 2.6, 1.2, 1.7, 0.9, "New images")
    box(ax, 4.9, 1.2, 1.9, 0.9, "Doctors rate\n1, 2 or 3")
    box(ax, 7.4, 1.2, 2.3, 0.9, "Reward model learns\nthe doctors' ratings")
    arrow(ax, 2.0, 1.65, 2.6, 1.65)
    arrow(ax, 4.3, 1.65, 4.9, 1.65)
    arrow(ax, 6.8, 1.65, 7.4, 1.65)
    arrow(ax, 8.55, 1.2, 1.05, 1.2, color=AQUA, rad=-0.2)
    ax.text(4.8, 0.62, "reward is used to train MINIM again (\"closed loop\")", ha="center", color=AQUA,
            fontsize=9.5, fontweight="bold")

    ax = axes[1]
    ax.text(0, 2.85, "What the released code does", fontsize=12, fontweight="bold", color=INK)
    box(ax, 0.1, 1.2, 1.9, 0.9, "Stable Diffusion 1.5\n(not MINIM's weights)", fc="#e6effa", ec=BLUE, bold=True)
    box(ax, 2.6, 1.2, 1.7, 0.9, "Make up to 10\nimages per prompt")
    box(ax, 4.9, 1.2, 1.9, 0.9, "Selector: same score\nfor every image")
    box(ax, 7.4, 1.2, 2.3, 0.9, "Policy picks the best\nof 3 candidates")
    arrow(ax, 2.0, 1.65, 2.6, 1.65)
    arrow(ax, 4.3, 1.65, 4.9, 1.65)
    arrow(ax, 6.8, 1.65, 7.4, 1.65)
    box(ax, 4.0, 0.05, 3.2, 0.62, "Doctors' ratings: a fixed spreadsheet,\ntrains the selector and the policy only",
        fc="#fdf0ea", ec=ORANGE)
    arrow(ax, 7.2, 0.42, 8.3, 1.2, color=ORANGE)
    arrow(ax, 5.6, 0.67, 5.85, 1.2, color=ORANGE)
    ax.plot([1.05, 1.05], [0.72, 1.18], color=MUTED, lw=1.2, ls=(0, (3, 3)))
    ax.text(1.05, 0.38, "image maker never updated\n(update path off, and crashes)", ha="center", color=INK2,
            fontsize=8.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_loop.png", dpi=220)
    plt.close(fig)


def fig_policy(r):
    d = r["policy_loss"]["network"]
    code = [d["MINIM_code_stage1"]["mean_score_by_rating"][str(g)] for g in (1, 2, 3)]
    paper = [d["paper_equation"]["mean_score_by_rating"][str(g)] for g in (1, 2, 3)]
    x = np.arange(3)
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    b1 = ax.bar(x - w / 2 - 0.01, code, w, color=ORANGE, label="MINIM's code (abs + log loss)")
    b2 = ax.bar(x + w / 2 + 0.01, paper, w, color=BLUE, label="Paper's equation (R - pi)")
    for xi, t in zip(x, [0.0, 0.5, 1.0]):
        ax.plot([xi - 0.42, xi + 0.42], [t, t], color=INK, lw=1.4, ls=(0, (4, 2)))
    ax.text(2.45, 1.0, "target", va="center", fontsize=8.8, color=INK)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}", ha="center",
                    fontsize=8.8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(["Doctor rating 1\n(low quality)", "Rating 2\n(good, wrong report)", "Rating 3\n(good, matches)"],
                       fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Average predicted score")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_policy.png", dpi=220)
    plt.close(fig)


def fig_loading():
    w = RESULTS / "t04_work"
    hu, theirs, mri = np.load(w / "ct_hu.npy"), np.load(w / "ct_theirs.npy"), np.load(w / "mri_theirs.npy")
    win = lambda h, c, wd: np.clip((h - (c - wd / 2)) / wd, 0, 1)
    fig, ax = plt.subplots(1, 4, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.35]})
    ax[0].imshow(win(hu, -600, 1500), cmap="gray")
    ax[0].set_title("Test CT slice\n(lung window, correct)", fontsize=9.5)
    ax[1].imshow(win(hu, 40, 400), cmap="gray")
    ax[1].set_title("Soft-tissue window\n(correct)", fontsize=9.5)
    ax[2].imshow(theirs, cmap="gray", vmin=0, vmax=1)
    ax[2].set_title("MINIM's loader output\n(same file: only 2 shades)", fontsize=9.5)
    for a in ax[:3]:
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_color(GRID)
    row = mri[256, 64:448]
    xs = np.linspace(0, 3000, len(row))
    ax[3].plot(xs, xs / 3000, color=MUTED, lw=2, ls=(0, (4, 2)), label="what it should be")
    ax[3].plot(xs, row, color=ORANGE, lw=2, label="MINIM's loader")
    ax[3].set_xlabel("MRI intensity in the file (0 to 3000)")
    ax[3].set_ylabel("Brightness after loading")
    ax[3].set_title("Test MRI volume (NIfTI)\n(a smooth ramp)", fontsize=9.5)
    ax[3].legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    ax[3].grid(color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_loading.png", dpi=220)
    plt.close(fig)


def fig_gate(r):
    g = r["quality_gate_on_real_images"]
    order = ["OCT", "Fundus", "CXR", "ChestCT", "BrainMRI", "BreastMRI"]
    labels = ["OCT", "Fundus", "Chest X-ray", "Chest CT", "Brain MRI", "Breast MRI"]
    pct = [100 * int(g[k]["passed"]) / g[k]["of"] for k in order]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    bars = ax.barh(labels[::-1], pct[::-1], color=BLUE, height=0.55)
    for b, k in zip(bars, order[::-1]):
        ax.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2, f"{g[k]['passed']} of {g[k]['of']}",
                va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Real example images that pass MINIM's quality gate (%)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_gate.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    fig_loop()
    r4 = load("t04_rlhf")
    if r4:
        fig_policy(r4)
        fig_loading()
        fig_gate(r4)
    print("figures in", OUT)
