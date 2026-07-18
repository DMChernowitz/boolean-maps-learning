import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FUNCS = [
    ("FALSE", 0b0000, 0), ("AND", 0b0001, 1), ("a & ~b", 0b0010, 1),
    ("a (proj)", 0b0011, 0), ("~a & b", 0b0100, 1), ("b (proj)", 0b0101, 0),
    ("XOR", 0b0110, 3), ("OR", 0b0111, 1), ("NOR", 0b1000, 1), ("XNOR", 0b1001, 3),
    ("~b", 0b1010, 0), ("b -> a", 0b1011, 1), ("~a", 0b1100, 0),
    ("a -> b", 0b1101, 1), ("NAND", 0b1110, 1), ("TRUE", 0b1111, 0),
]

PSI = 0b0001  # true map = AND

def epsilon(f, psi, rows=4):
    return sum(1 for k in range(rows) if ((f >> k) & 1) == ((psi >> k) & 1)) / rows

rows = []
for name, f, C in FUNCS:
    eps = epsilon(f, PSI)
    rows.append((name, f, C, eps))

rows.sort(key=lambda r: r[3])  # sort ascending by epsilon

names = [r[0] for r in rows]
Cs = np.array([r[2] for r in rows])
epss = np.array([r[3] for r in rows])
rank = np.arange(1, len(rows) + 1)

# --- style constants (dataviz skill: light surface, blue sequential/series-1) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})

# Chart 1: epsilon_j vs rank (sorted ascending)
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.plot(rank, epss, color=BLUE, lw=2, marker="o", markersize=8, zorder=3)
ax.set_xticks(rank)
ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
ax.set_ylabel(r"$\epsilon_j$ (accuracy vs. $\psi=$AND, uniform $P$)")
ax.set_ylim(-0.05, 1.05)
ax.grid(axis="y", color=GRID, lw=1, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.set_title(r"$\epsilon_j$ sorted ascending ($\psi=$AND)", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig("figures/epsilon_sorted.png", facecolor=SURFACE)
plt.close(fig)

# Chart 2: Boltzmann prior p_j^Boltz(beta) over the SAME rank order, for a few betas
betas = [0.5, 1.0, 2.0]
colors = [BLUE, AQUA, YELLOW]
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for beta, color in zip(betas, colors):
    w = np.exp(-beta * Cs)
    p = w / w.sum()
    ax.plot(rank, p, color=color, lw=2, marker="o", markersize=8, zorder=3,
            label=rf"$\beta={beta}$")
ax.set_xticks(rank)
ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
ax.set_ylabel(r"$p_j^{\mathrm{Boltz}}(\beta) \propto e^{-\beta C_j}$")
ax.grid(axis="y", color=GRID, lw=1, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.legend(frameon=False, labelcolor=INK)
ax.set_title(r"Boltzmann prior over complexity, same rank order as $\epsilon_j$", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig("figures/epsilon_boltzmann.png", facecolor=SURFACE)
plt.close(fig)

# markdown table
print("| rank | $\\phi_j$ | truth table | $C_j$ | $\\epsilon_j$ |")
print("|---|---|---|---|---|")
for i, (name, f, C, eps) in enumerate(rows, start=1):
    print(f"| {i} | {name} | {format(f,'04b')} | {C} | {eps:.2f} |")
