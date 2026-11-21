"""Final evaluation of every method against every attack (Tables I and II).

Methods: the baselines, the shielded RL defender without memory and the unshielded
ablation (03_shielded_rl.py), and on the unseen graphs the memory + fallback defender
and its game-trained version. Attacks: the scripted attacks plus, for each method, an
RL attacker trained against that method itself (02_rl_attackers.py for the baselines;
trained here for the learned defenders, warm-started from the LF attacker).

    python experiments/06_evaluate.py               # 100 unseen graphs per F -> results/evaluation/ (paper)
    python experiments/06_evaluate.py --benchmark   # 10 benchmark graphs     -> results/evaluation_benchmark/

Best-response attackers that already exist are reused; missing ones are trained.
"""

import argparse
import json
import os
from multiprocessing import get_context

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from advagent import bench
from advagent import checkpoints as C
from advagent import plots
from advagent import learn as Lr
from advagent import torchsim as ts

UNSEEN = os.environ.get("ADV_EVAL_BENCHMARK") != "1"   # set by --benchmark; read again in worker processes
OUT = C.EVALUATION if UNSEEN else C.EVALUATION_BENCHMARK
SEARCH = (C.EVALUATION, C.EVALUATION_BENCHMARK) if UNSEEN else (C.EVALUATION_BENCHMARK,)
BASE = C.BASELINES
LEARNED = ["Shielded RL", "Unshielded RL"]
if UNSEEN:
    LEARNED = LEARNED + C.OURS


def instances(F):
    return Lr.train_pool(F, 100, offset=300_000) if UNSEEN else Lr.eval_instances(F)
BR = "RL attacker (best response)"


def factory(name):
    if name in BASE:
        return lambda: ts.BASELINES[name]()
    pol = C.defender(name)
    return lambda: pol


def br_job(job):
    """Train a best-response attacker against a learned defender."""
    name, F, iters = job
    torch.set_num_threads(3)
    rule = C.defender(name)
    init = C.rl_attacker("LF", F)   # warm start from the LF attacker
    pol = init.train() if init is not None else Lr.AttackerPolicy()
    pol, curve = Lr.train_attacker(lambda: rule, Lr.train_pool(F, 512), iters=iters, policy=pol, seed=7)
    (OUT / "attackers").mkdir(parents=True, exist_ok=True)
    torch.save(pol.state_dict(), OUT / "attackers" / f"{C.slug(name)}_F{F}.pt")
    return name, F


def eval_job(job):
    name, F = job
    torch.set_num_threads(3)
    insts = instances(F)
    fac = factory(name)
    res = {}
    with torch.no_grad():
        for att, af in bench.SCRIPTED.items():
            res[(name, F, att)] = bench.evaluate(fac, insts, lambda af=af, F=F: af(F))
        pol = C.best_response_attacker(name, F, SEARCH)
        if pol is not None:
            runs = [bench.evaluate(fac, insts, lambda dr=dr: Lr.LearnedAttack(pol, direc=dr)) for dr in (1, -1)]
            pick = np.argmax([runs[0]["error"], runs[1]["error"]], 0)
            res[(name, F, BR)] = {k: np.where(pick == 0, runs[0][k], runs[1][k]) for k in runs[0]}
    return res


def evaluate_all(methods):
    res = {}
    with get_context("spawn").Pool(4 if UNSEEN else 7) as pool:   # memory defenders keep per-step memory: RAM
        for r in pool.map(eval_job, [(n, F) for F in (1, 2) for n in methods], chunksize=1):
            res.update(r)
    return res


def uw(methods):
    out = {}
    for K in (2, 5):
        for name in methods:
            out[(name, K)] = float(bench.evaluate(factory(name), [bench.uw_instance(K)], lambda: None, T=4000)["error"][0])
    return out


def attack_list(res, name, F):
    return [a for a in list(bench.SCRIPTED) + [BR] if (name, F, a) in res]


def summarize(res, methods):
    rows = {}
    for name in methods:
        for F in (1, 2):
            atts = [a for a in attack_list(res, name, F) if a != "none"]
            errs = {a: res[(name, F, a)]["error"].mean() for a in atts}
            worst = max(errs, key=errs.get)
            unsafe = [a for a in attack_list(res, name, F) if bench.flag(res[(name, F, a)]) == "unsafe"]
            nocons = [a for a in attack_list(res, name, F) if bench.flag(res[(name, F, a)]) == "no consensus"]
            rows[(name, F)] = {"none": float(res[(name, F, "none")]["error"].mean()), "mean": float(np.mean(list(errs.values()))),
                               "worst": float(errs[worst]), "worst_attack": worst, "br": float(errs.get(BR, np.nan)),
                               "unsafe": unsafe, "no_consensus": nocons}
    return rows


def table(res, rows, methods, uwr):
    lines = ["# Final evaluation", "",
             ("100 unseen random 20-node graphs per F (seeds disjoint from training and from the benchmark), T = 2000. "
              if UNSEEN else "Benchmark graphs (10 random 20-node graphs per F, T = 2000). ") +
             f"'{BR}' is an RL attacker trained against that specific method (02_rl_attackers.py for the baselines, 06_evaluate.py for the "
             "learned defenders); it picks the worse push direction per instance. Worst case and mean are over all "
             "attacks except 'none'. Unsafe: some run's hull violation (t ≥ T/2) or α = 0 range excess above 1e−3. "
             "No consensus: final spread above 1.", ""]
    for F in (1, 2):
        lines += [f"## Summary, F = {F}", "",
                  "| Method | No attack | Mean over attacks | Worst case (attack) | Best-response RL attacker | Safe under every attack? |",
                  "|---|---|---|---|---|---|"]
        for name in methods:
            r = rows[(name, F)]
            safe = "yes" if not r["unsafe"] and not r["no_consensus"] else (
                ("**no**: unsafe under " + ", ".join(r["unsafe"])) if r["unsafe"] else "no consensus under " + ", ".join(r["no_consensus"]))
            br = "—" if np.isnan(r["br"]) else f"{r['br']:.3f}"
            lines.append(f"| {name} | {r['none']:.3f} | {r['mean']:.3f} | {r['worst']:.3f} ({r['worst_attack']}) | {br} | {safe} |")
        lines.append("")
    for F in (1, 2):
        atts = list(bench.SCRIPTED) + [BR]
        lines += [f"## Full grid, F = {F}: mean error", "", "| Method | " + " | ".join(atts) + " |", "|---" * (len(atts) + 1) + "|"]
        for name in methods:
            cells = []
            for a in atts:
                m = res.get((name, F, a))
                cells.append("—" if m is None else f"{m['error'].mean():.3f}{bench.MARK[bench.flag(m)]}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## U–W graph (Ex. 8.2, b = 4, no adversaries): error vs the |T|/n·b = b/4 = 1 bound", "",
              "| Method | K = 2 | K = 5 |", "|---|---|---|"]
    lines += [f"| {n} | {uwr[(n, 2)]:.3f} | {uwr[(n, 5)]:.3f} |" for n in methods]
    return "\n".join(lines) + "\n"


def robustness_plot(rows, methods):
    safe_methods = [m for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, F in zip(axes, (1, 2)):
        names = safe_methods
        x = np.arange(len(names))
        none = [rows[(n, F)]["none"] for n in names]
        mean = [rows[(n, F)]["mean"] for n in names]
        worst = [rows[(n, F)]["worst"] for n in names]
        w = 0.26
        ax.bar(x - w, none, w, color="#9ec5ef", label="no attack")
        ax.bar(x, mean, w, color="#2a78d6", label="mean over attacks")
        ax.bar(x + w, worst, w, color="#0b3e7a", label="worst case")
        for i, n in enumerate(names):
            if rows[(n, F)]["unsafe"] or rows[(n, F)]["no_consensus"]:
                ax.text(i, max(worst[i], mean[i]) * 1.15, "✕", ha="center", va="bottom", color="#d6311f", fontsize=11)
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 300)
        for i, v in enumerate(none):
            if v < 1e-2:   # DGD without attack is exact; mark it instead of stretching the axis
                ax.text(i - w, 1.2e-2, "≈0", ha="center", va="bottom", fontsize=7, color="#555")
        ax.set_xticks(x, names, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("mean error |x̄(T) − x*| (log)")
        ax.set_title(f"F = {F}", fontsize=10)
        ax.grid(axis="y", color="#e6e5df", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper center", ncol=3)
    fig.suptitle("Accuracy and robustness of every method (✕ = unsafe or no consensus under some attack)"
                 + (", 100 unseen graphs per F" if UNSEEN else ""), fontsize=11)
    fig.savefig(OUT / "robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--br-iters", type=int, default=200)
    ap.add_argument("--skip-br", action="store_true")
    ap.add_argument("--benchmark", action="store_true", help="the 10 benchmark graphs instead of 100 unseen ones")
    args = ap.parse_args()
    if args.benchmark and UNSEEN:   # re-run with the environment set, so worker processes see it too
        import subprocess
        import sys
        env = dict(os.environ, ADV_EVAL_BENCHMARK="1")
        sys.exit(subprocess.call([sys.executable] + sys.argv, env=env))
    OUT.mkdir(parents=True, exist_ok=True)
    methods = BASE + LEARNED
    if not args.skip_br:
        todo = [n for n in LEARNED if not all(C.best_response_path(n, F, SEARCH).exists() for F in (1, 2))] if UNSEEN else LEARNED
        with get_context("spawn").Pool(4 if UNSEEN else 8) as pool:
            pool.map(br_job, [(n, F, args.br_iters) for n in todo for F in (1, 2)], chunksize=1)
    torch.set_num_threads(8)
    res = evaluate_all(methods)
    rows = summarize(res, methods)
    uwr = uw(methods)
    md = table(res, rows, methods, uwr)
    (OUT / "table.md").write_text(md)
    (OUT / "results.json").write_text(json.dumps({
        "grid": {f"{k[0]}|F{k[1]}|{k[2]}": {m: v.tolist() for m, v in d.items()} for k, d in res.items()},
        "summary": {f"{k[0]}|F{k[1]}": v for k, v in rows.items()},
        "uw": {f"{k[0]}|K{k[1]}": v for k, v in uwr.items()}}, indent=1))
    plots.heatmap(res, {m: None for m in methods}, OUT / "heatmap.png", "Every method against every attack"
               + (" (100 unseen graphs per F)" if UNSEEN else ""))
    robustness_plot(rows, methods)
    print(md)
