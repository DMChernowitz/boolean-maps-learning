import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

n, m = 2, 1
rows = 1 << n
counts = [math.comb(rows, d) * (2**m - 1)**d for d in range(rows + 1)]
eps = [(rows - d) / rows for d in range(rows + 1)]

assert sum(counts) == 2**(m * rows)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})

fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
bars = ax.bar([f"{e:.2f}" for e in eps], counts, color=BLUE, width=0.6, zorder=3)
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, str(c),
            ha="center", va="bottom", fontsize=10, color=INK)
ax.set_xlabel(r"$\epsilon_j$")
ax.set_ylabel(r"count of $\phi_j$ at this $\epsilon_j$")
ax.grid(axis="y", color=GRID, lw=1, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.set_title(r"$\binom{4}{d}$: hypotheses per accuracy level ($n=2,m=1$)", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig("figures/epsilon_counts.png", facecolor=SURFACE)
plt.close(fig)
print("counts:", counts, "sum:", sum(counts))
