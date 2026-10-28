"""Phase 1: reproduce the dynamics results of Sundaram & Gharesifard [1].

Writes figures and a summary to results/phase1/. Run from the repo root:
    python3 experiments/phase1_reproduce.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from advagent import attacks as A
from advagent import dynamics as d
from advagent import graphs as g
from advagent.functions import LocalFunctions

OUT = Path(__file__).resolve().parents[1] / "results" / "phase1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, BAND = "#1f1f1e", "#6b6a63", "#e6e5df", "#f1f0ea"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def hull_band(ax, hull, label="minimizer hull"):
    ax.axhspan(*hull, color=BAND, zorder=0, label=label)


def prop_4_3():
    """One stubborn adversary captures DGD; LF stays in the hull."""
    rng = np.random.default_rng(1)
    n = 10
    adj = g.random_robust(n, 3, rng)
    a = rng.uniform(-1, 1, n)
    f = LocalFunctions.quadratic(a)
    kw = dict(T=3000, alpha=d.power_steps(0.5, 0.6), adversaries=[n - 1], attack=A.Fixed(50.0))
    runs = {rule.name: d.simulate(adj, f, a.copy(), rule, **kw) for rule in (d.DGD(), d.LF(1))}

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    hull_band(ax, runs["LF"].hull)
    ax.plot(runs["DGD"].mean, color=ORANGE, lw=1.5, label="DGD (no defence)")
    ax.plot(runs["LF"].mean, color=BLUE, lw=1.5, label="LF, F = 1")
    ax.axhline(50, color=MUTED, lw=0.8, ls=":", label="adversary's value (50)")
    ax.set(xlabel="time step t", ylabel="mean regular state",
           title="One fixed adversary drags DGD to its value; LF stays in the hull")
    ax.legend(loc="center right")
    fig.savefig(OUT / "prop4_3_capture.png")
    plt.close(fig)
    return {k: {"final_mean": float(r.mean[-1]), "hull": r.hull, "x_star": r.x_star} for k, r in runs.items()}


def ex_7_2():
    """Switching malicious node keeps the regular nodes oscillating forever."""
    adj = g.complete(5)
    f = LocalFunctions.quadratic([0, 0, 0, 9, 0], L=100)
    att = A.Switching(low=[0, 1, 2], odd=3)
    T = 3000
    r = d.simulate(adj, f, np.array([0, 0, 0, 9, 0.0]), d.LF(1), T, d.power_steps(0.5, 0.6),
                   adversaries=[4], attack=att)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    t = np.arange(T + 1)
    ax.plot(t, r.X[:, 0], color=BLUE, lw=1.5, label="regular v1-v3 (f = x²)")
    ax.plot(t, r.X[:, 3], color=ORANGE, lw=1.5, ls="--", label="regular v4 (f = (x-9)²)")
    ax.plot(t, r.X[:, 4], color=MUTED, lw=0.8, ls=":", label="malicious v5")
    ax.set(xlabel="time step t", ylabel="state", ylim=(-0.3, 4.5),
           title="Example 7.2: a switching malicious node prevents convergence")
    ax.legend(loc="upper right")
    fig.savefig(OUT / "ex7_2_oscillation.png")
    plt.close(fig)
    tail = r.mean[T // 2:]
    return {"switches": len(att.switches), "late_mean_range": [float(tail.min()), float(tail.max())],
            "hull_violation_after_t10": r.hull_violation(10)}


def ex_8_2():
    """LF output is b/4 away from x* on the U-W graph, for every K."""
    b = 4.0
    out = {}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for K, (rule, color) in [(5, (d.DGD(), ORANGE)), (5, (d.LF(1), BLUE))]:
        adj, W, U = g.uw_graph(K)
        a = np.zeros(adj.shape[0])
        a[U] = b
        x0 = np.random.default_rng(0).uniform(-2, 6, a.size)
        r = d.simulate(adj, LocalFunctions.quadratic(a, L=100), x0, rule, 4000, d.power_steps(0.5, 0.6))
        # Worst node, not the mean: with alpha_0 = 0.5 the first DGD step puts the
        # mean exactly on x*, and the remaining error is disagreement between nodes.
        ax.plot(np.abs(r.XR - r.x_star).max(axis=1), color=color, lw=1.5, label=f"{rule.name} (no adversaries)")
        out[rule.name] = {"K": K, "x_star": r.x_star, "final_error": r.error()}
    for K in (2, 10, 20):
        adj, W, U = g.uw_graph(K)
        a = np.zeros(adj.shape[0])
        a[U] = b
        r = d.simulate(adj, LocalFunctions.quadratic(a, L=100), a.copy(), d.LF(1), 4000, d.power_steps(0.5, 0.6))
        out[f"LF_K{K}"] = r.error()
    ax.axhline(b / 4, color=MUTED, lw=0.8, ls=":", label="bound |T|/n·|b-a| = b/4")
    ax.set(xlabel="time step t", ylabel="max_i |x_i - x*|", yscale="log", ylim=(1e-2, 5),
           title="Example 8.2: secure LF pays the b/4 price even with no adversaries")
    ax.legend(loc="lower left")
    fig.savefig(OUT / "ex8_2_lower_bound.png")
    plt.close(fig)
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"prop_4_3": prop_4_3(), "ex_7_2": ex_7_2(), "ex_8_2": ex_8_2()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
