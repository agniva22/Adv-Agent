"""Phase 2: compare every method against the scripted attacks on random graphs.

For each seed: a random 3-robust 20-node graph, a random 1-local adversary set
(up to 3 nodes), random quadratics with minimizers in [-5, 5], and random
initial states. Writes results/phase2/{table.md, raw.json, errors.png}.
    PYTHONPATH=. python3 experiments/phase2_baselines.py
"""

import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from advagent import attacks as A
from advagent import baselines as B
from advagent import dynamics as d
from advagent import graphs as g
from advagent.functions import LocalFunctions

OUT = Path(__file__).resolve().parents[1] / "results" / "phase2"
F, N, T, SEEDS = 1, 20, 2000, 10
ALPHA = d.power_steps(0.5, 0.6)

METHODS = {
    "DGD": lambda: d.DGD(),
    "LF": lambda: d.LF(F),
    "LF-Metropolis": lambda: B.LFMetropolis(F),
    "Su-Vaidya": lambda: B.SuVaidya(F),
    "Q-consensus": lambda: B.QConsensus(scale=10.0),
    "Q-consensus (fixed beta)": lambda: B.QConsensus(beta1=0.0, scale=10.0),
    "DRC-RSSG": lambda: B.DRCRSSG(F),
}
ATTACKS = {
    "none": lambda: None,
    "fixed +50": lambda: A.Fixed(50.0),
    "random Byzantine": lambda: A.UniformRandom(-50, 50, byzantine=True),
    "farthest evader": lambda: A.FarthestEvader(+1, 0.9),
}


def instance(seed):
    rng = np.random.default_rng(seed)
    adj = g.random_robust(N, 2 * F + 1, rng, extra=1)
    adv = g.random_f_local_set(adj, F, 3, rng)
    a = rng.uniform(-5, 5, N)
    return adj, adv, LocalFunctions.quadratic(a, L=100), rng.uniform(-5, 5, N)


def run(job):
    seed, method, attack = job
    adj, adv, f, x0 = instance(seed)
    att = ATTACKS[attack]()
    advs = [] if att is None else adv
    r = d.simulate(adj, f, x0, METHODS[method](), T, ALPHA, adversaries=advs, attack=att, seed=seed)
    # Safety in the pure-consensus setting of [10]: no gradients, states must
    # stay in the regular nodes' initial range.
    s = d.simulate(adj, f, x0, METHODS[method](), 300, lambda t: 0.0, adversaries=advs, attack=att, seed=seed)
    reg0 = x0[s.regular]
    excess = max(0.0, reg0.min() - s.XR.min(), s.XR.max() - reg0.max())
    return {"seed": seed, "method": method, "attack": attack, "error": r.error(),
            "spread": float(r.spread[-1]), "hull_violation": r.hull_violation(T // 2),
            "consensus_range_excess": float(excess)}


def table(rows):
    def cell(vals, fmt="{:.3f}"):
        v = np.array(vals)
        return f"{fmt.format(v.mean())} ± {fmt.format(v.std())}"

    lines = []
    for attack in ATTACKS:
        lines += [f"\n### Attack: {attack}\n",
                  "| Method | Error \\|x̄ − x*\\| | Final spread D(T) | Max hull violation (t ≥ T/2) | Safety: excess over initial range (α = 0) |",
                  "|---|---|---|---|---|"]
        for m in METHODS:
            sel = [r for r in rows if r["method"] == m and r["attack"] == attack]
            lines.append(f"| {m} | {cell([r['error'] for r in sel])} | {cell([r['spread'] for r in sel])} | "
                         f"{max(r['hull_violation'] for r in sel):.3f} | {max(r['consensus_range_excess'] for r in sel):.3f} |")
    return "\n".join(lines)


def figure(rows):
    colors = {"DGD": "#eb6834", "LF": "#2a78d6", "LF-Metropolis": "#1baf7a", "Su-Vaidya": "#eda100",
              "Q-consensus": "#e87ba4", "Q-consensus (fixed beta)": "#b3477a", "DRC-RSSG": "#008300"}
    fig, axes = plt.subplots(1, len(ATTACKS), figsize=(12, 3.6), sharey=True)
    for ax, attack in zip(axes, ATTACKS):
        for k, m in enumerate(METHODS):
            v = np.maximum(1e-3, [r["error"] for r in rows if r["method"] == m and r["attack"] == attack])
            med = np.median(v)
            ax.errorbar(k, med, yerr=[[med - v.min()], [v.max() - med]], fmt="o", ms=5,
                        color=colors[m], elinewidth=1.2, capsize=0)
        ax.set_title(attack, fontsize=10)
        ax.set_xticks(range(len(METHODS)), METHODS, rotation=60, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(8e-4, 100)
        ax.grid(axis="y", color="#e6e5df", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("final error |x̄ − x*|\n(median, min–max over seeds; floored at 1e-3)")
    fig.suptitle("Error of each method under each scripted attack (10 random 20-node 3-robust graphs, F = 1)", fontsize=11)
    fig.savefig(OUT / "errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, m, a) for s in range(SEEDS) for m in METHODS for a in ATTACKS]
    with Pool() as pool:
        rows = pool.map(run, jobs)
    (OUT / "raw.json").write_text(json.dumps(rows, indent=1))
    md = table(rows)
    (OUT / "table.md").write_text(
        f"# Phase 2 baseline comparison\n\n{SEEDS} seeds; random {N}-node 3-robust graphs; F = {F}; "
        f"T = {T}; alpha_t = 0.5/(t+1)^0.6. Mean ± std over seeds; violations are worst case.\n{md}\n")
    figure(rows)
    print(md)
