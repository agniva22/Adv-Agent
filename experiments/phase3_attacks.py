"""Phase 3: every method against every scripted attack.

Main grid: for F = 1 and F = 2, 10 random (2F+1)-robust 20-node graphs with a
random F-local adversary set of up to 3F nodes, random quadratics with
minimizers in [-5, 5], and random initial states. Plus the two special graphs
of [1]: the Ex. 7.2 complete graph with its switching attack, and the U-W
graph of Ex. 8.2 (no adversaries; the price of security). Writes
results/phase3/{table.md, raw.json, heatmap.png, switching.png}.
    PYTHONPATH=. python3 experiments/phase3_attacks.py
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

OUT = Path(__file__).resolve().parents[1] / "results" / "phase3"
N, T, SEEDS = 20, 2000, 10
ALPHA = d.power_steps(0.5, 0.6)
UNSAFE_TOL, SPREAD_TOL = 1e-3, 1.0

METHODS = {
    "DGD": lambda F: d.DGD(),
    "LF": lambda F: d.LF(F),
    "LF-Metropolis": lambda F: B.LFMetropolis(F),
    "Su-Vaidya": lambda F: B.SuVaidya(F),
    "Q-consensus": lambda F: B.QConsensus(scale=10.0),
    "Q-consensus (fixed beta)": lambda F: B.QConsensus(beta1=0.0, scale=10.0),
    "DRC-RSSG": lambda F: B.DRCRSSG(F),
}
# Each attack gets the schedule it must mimic; the alpha = 0 safety run passes ZERO.
ATTACKS = {
    "none": lambda F, al: None,
    "fixed +50": lambda F, al: A.Fixed(50.0),
    "random Byzantine": lambda F, al: A.UniformRandom(-50, 50, byzantine=True),
    "farthest evader": lambda F, al: A.FarthestEvader(+1, 0.9),
    "stealthy (fake 5)": lambda F, al: A.Stealthy(5.0, al),
    "stealthy (fake 50)": lambda F, al: A.Stealthy(50.0, al),
    "trim-hugger": lambda F, al: A.TrimHugger(F, +1),
    "switching": lambda F, al: A.StallSwitcher(A.TrimHugger(F, +1), A.TrimHugger(F, -1)),
}
ZERO = lambda t: 0.0


def instance(seed, F):
    rng = np.random.default_rng(1000 * F + seed)
    adj = g.random_robust(N, 2 * F + 1, rng, extra=1)
    adv = g.random_f_local_set(adj, F, 3 * F, rng)
    a = rng.uniform(-5, 5, N)
    return adj, adv, LocalFunctions.quadratic(a, L=100), rng.uniform(-5, 5, N)


def run(job):
    F, seed, method, attack = job
    adj, adv, f, x0 = instance(seed, F)
    att = ATTACKS[attack](F, ALPHA)
    advs = [] if att is None else adv
    r = d.simulate(adj, f, x0, METHODS[method](F), T, ALPHA, adversaries=advs, attack=att, seed=seed)
    # Safety in the pure-consensus setting of [10]: no gradients, states must
    # stay in the regular nodes' initial range.
    att0 = ATTACKS[attack](F, ZERO)
    s = d.simulate(adj, f, x0, METHODS[method](F), 300, ZERO, adversaries=advs, attack=att0, seed=seed)
    reg0 = x0[s.regular]
    excess = max(0.0, reg0.min() - s.XR.min(), s.XR.max() - reg0.max())
    return {"F": F, "seed": seed, "method": method, "attack": attack, "n_adv": len(advs),
            "error": r.error(), "spread": float(r.spread[-1]), "hull_violation": r.hull_violation(T // 2),
            "oscillation": r.oscillation(T // 2), "consensus_range_excess": float(excess)}


def ex_7_2(method):
    """The original switching attack of [1, Ex. 7.2] against each method (F = 1)."""
    adj = g.complete(5)
    f = LocalFunctions.quadratic([0, 0, 0, 9, 0], L=100)
    att = A.Switching(low=[0, 1, 2], odd=3)
    r = d.simulate(adj, f, np.array([0, 0, 0, 9, 0.0]), METHODS[method](1), 3000, ALPHA,
                   adversaries=[4], attack=att)
    return {"method": method, "error": r.error(), "oscillation": r.oscillation(1500),
            "switches": len(att.switches), "hull_violation": r.hull_violation(10)}


def uw(method, K=5, b=4.0):
    """Ex. 8.2: no adversaries; a secure method should still pay about b/4 = |T|/n * b."""
    adj, W, U = g.uw_graph(K)
    a = np.zeros(adj.shape[0])
    a[U] = b
    r = d.simulate(adj, LocalFunctions.quadratic(a, L=100), a.copy(), METHODS[method](1), 4000, ALPHA)
    return {"method": method, "error": r.error(), "bound": b / 4}


def unsafe(r):
    return r["hull_violation"] > UNSAFE_TOL or r["consensus_range_excess"] > UNSAFE_TOL


def flag(sel):
    """'unsafe' if any run left the safe set, 'no consensus' if any run did not agree."""
    if any(unsafe(r) for r in sel):
        return "unsafe"
    if any(r["spread"] > SPREAD_TOL for r in sel):
        return "no consensus"
    return ""


def table(rows, ex72, uwr):
    def cell(vals):
        v = np.array(vals)
        return f"{v.mean():.3f} ± {v.std():.3f}"

    lines = []
    for F in (1, 2):
        lines += [f"\n## F = {F}\n", "### Error |x̄ − x*| (mean ± std over seeds); ⚠ = some run unsafe, ∅ = some run without consensus\n",
                  "| Method | " + " | ".join(ATTACKS) + " |", "|---" * (len(ATTACKS) + 1) + "|"]
        for m in METHODS:
            cells = []
            for at in ATTACKS:
                sel = [r for r in rows if r["F"] == F and r["method"] == m and r["attack"] == at]
                mark = {"unsafe": " ⚠", "no consensus": " ∅", "": ""}[flag(sel)]
                cells.append(f"{np.mean([r['error'] for r in sel]):.3f}{mark}")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        for at in ATTACKS:
            lines += [f"\n### F = {F}, attack: {at}\n",
                      "| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |",
                      "|---|---|---|---|---|---|"]
            for m in METHODS:
                sel = [r for r in rows if r["F"] == F and r["method"] == m and r["attack"] == at]
                lines.append(f"| {m} | {cell([r['error'] for r in sel])} | {cell([r['spread'] for r in sel])} | "
                             f"{cell([r['oscillation'] for r in sel])} | {max(r['hull_violation'] for r in sel):.3f} | "
                             f"{max(r['consensus_range_excess'] for r in sel):.3f} |")
    lines += ["\n## Ex. 7.2 graph (K5, switching attack of [1])\n",
              "| Method | Final error | Oscillation (TV, t ≥ 1500) | Attacker switches | Hull violation (t ≥ 10) |", "|---|---|---|---|---|"]
    lines += [f"| {r['method']} | {r['error']:.3f} | {r['oscillation']:.3f} | {r['switches']} | {r['hull_violation']:.3f} |" for r in ex72]
    lines += ["\n## U–W graph (Ex. 8.2, K = 5, b = 4, no adversaries)\n",
              "| Method | Error | Bound \\|T\\|/n·b = b/4 |", "|---|---|---|"]
    lines += [f"| {r['method']} | {r['error']:.3f} | {r['bound']:.2f} |" for r in uwr]
    return "\n".join(lines)


def heatmap(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    methods, attacks = list(METHODS), list(ATTACKS)
    norm = matplotlib.colors.LogNorm(1e-2, 50)
    for ax, F in zip(axes, (1, 2)):
        E = np.zeros((len(methods), len(attacks)))
        for i, m in enumerate(methods):
            for j, at in enumerate(attacks):
                sel = [r for r in rows if r["F"] == F and r["method"] == m and r["attack"] == at]
                E[i, j] = np.mean([r["error"] for r in sel])
                f = flag(sel)
                light = norm(max(E[i, j], 1e-2)) < 0.55
                ax.text(j, i, f"{E[i, j]:.2f}" if E[i, j] < 10 else f"{E[i, j]:.0f}", ha="center", va="center",
                        fontsize=8, color="#1f1f1f" if light else "white")
                if f:
                    ax.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False,
                                               ec="#d6311f" if f == "unsafe" else "#1f1f1f", lw=1.6,
                                               ls="-" if f == "unsafe" else "--"))
        im = ax.imshow(np.maximum(E, 1e-2), cmap="Blues", norm=norm, aspect="auto")
        ax.set_xticks(range(len(attacks)), attacks, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(methods)), methods, fontsize=8)
        ax.set_title(f"F = {F}", fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("mean error |x̄ − x*| over 10 graphs (log)")
    handles = [plt.Rectangle((0, 0), 1, 1, fill=False, ec="#d6311f", lw=1.6, label="unsafe in some run"),
               plt.Rectangle((0, 0), 1, 1, fill=False, ec="#1f1f1f", lw=1.6, ls="--", label="no consensus in some run")]
    fig.legend(handles=handles, loc="upper left", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.02, -0.08))
    fig.suptitle("Phase 3: error of each method under each scripted attack (random 20-node graphs)", fontsize=11)
    fig.savefig(OUT / "heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def switching_figure():
    """One graph, F = 1: the regular mean under the switching attack, for each safe method."""
    adj, adv, f, x0 = instance(0, 1)
    colors = {"LF": "#2a78d6", "LF-Metropolis": "#1baf7a", "Su-Vaidya": "#eda100"}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for m, c in colors.items():
        att = ATTACKS["switching"](1, ALPHA)
        r = d.simulate(adj, f, x0, METHODS[m](1), T, ALPHA, adversaries=adv, attack=att)
        ax.plot(r.mean - r.x_star, color=c, lw=1.4, label=f"{m} ({len(att.switches)} switches)")
    ax.axhline(0, color="#8a8a8a", lw=0.8, ls=":")
    ax.set(xlabel="time step t", ylabel="x̄(t) − x*", ylim=(-1.5, 1.5),
           title="Switching attack on a random graph: the mean never settles")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "switching.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(F, s, m, a) for F in (1, 2) for s in range(SEEDS) for m in METHODS for a in ATTACKS]
    with Pool() as pool:
        rows = pool.map(run, jobs)
        ex72 = pool.map(ex_7_2, list(METHODS))
        uwr = pool.map(uw, list(METHODS))
    (OUT / "raw.json").write_text(json.dumps({"grid": rows, "ex7_2": ex72, "uw": uwr}, indent=1))
    md = table(rows, ex72, uwr)
    (OUT / "table.md").write_text(
        f"# Phase 3 scripted-attack benchmark\n\n{SEEDS} seeds per F; random {N}-node (2F+1)-robust graphs; "
        f"up to 3F adversaries (F-local); T = {T}; alpha_t = 0.5/(t+1)^0.6. "
        f"Unsafe = hull violation or excess over the initial range above {UNSAFE_TOL}; "
        f"no consensus = final spread above {SPREAD_TOL}.\n{md}\n")
    heatmap(rows)
    switching_figure()
    print(md.split("\n## F = 2")[0])
