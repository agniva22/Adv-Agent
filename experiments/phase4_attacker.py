"""Phase 4: an RL attacker (proposal Formulation 5A) against every defence.

For each defence and F, PPO trains one attacker policy on random training
graphs (seeds disjoint from the benchmark), then it is evaluated on the exact
Phase 3 benchmark instances, T = 2000. The attacker may choose its push
direction per instance (it is omniscient), so we run both and keep the worse
error for the defence. Two oscillation attackers are also trained: on random
graphs, and on the Ex. 7.2 graph of [1], to see whether PPO rediscovers the
switching attack. Writes results/phase4/{table.md, results.json, curves.png,
attackers/*.pt}.
    PYTHONPATH=. python3 experiments/phase4_attacker.py [--iters N] [--procs P]
"""

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from advagent import attacks as A
from advagent import dynamics as d
from advagent import graphs as g
from advagent import learn as Lr
from advagent import torchsim as ts
from advagent.functions import LocalFunctions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase4"
T = 2000
K_OSC = 50   # an oscillation attacker decides every 50 steps
F1 = ["DGD", "LF", "LF-Metropolis", "Su-Vaidya", "Q-consensus", "Q-consensus (fixed beta)", "DRC-RSSG"]
F2 = ["LF", "LF-Metropolis", "Su-Vaidya", "DRC-RSSG"]


def ex72_instance():
    adj = g.complete(5)
    return {"adj": adj, "adv": [4], "a": np.array([0, 0, 0, 9, 0.0]), "x0": np.array([0, 0, 0, 9, 0.0]), "F": 1}


def tag(method, F, objective, graph="random"):
    return f"{method.replace(' ', '_').replace('(', '').replace(')', '')}_F{F}_{objective}_{graph}"


def train_job(job):
    method, F, objective, graph, iters, seed = job
    torch.set_num_threads(3)
    pool = [ex72_instance()] if graph == "ex72" else Lr.train_pool(F, 512)
    lines = []
    k = K_OSC if objective == "oscillation" else 10
    pol, lr = Lr.AttackerPolicy(), 3e-4
    if objective == "oscillation":
        # Low exploration noise: otherwise noise alone earns the reward and the
        # mean action never learns to switch (found in the pilot run).
        with torch.no_grad():
            pol.log_std.fill_(-2.0)
        lr = 1e-3
    pol, curve = Lr.train_attacker(ts.BASELINES[method], pool, T=T, k=k, iters=iters, objective=objective,
                                   seed=seed, policy=pol, lr=lr, log=lines.append)
    (OUT / "attackers").mkdir(parents=True, exist_ok=True)
    torch.save(pol.state_dict(), OUT / "attackers" / f"{tag(method, F, objective, graph)}.pt")
    res = evaluate(pol, method, F, objective, graph)
    res.update({"method": method, "F": F, "objective": objective, "graph": graph, "curve": curve, "log": lines})
    return res


@torch.no_grad()
def evaluate(pol, method, F, objective, graph):
    insts = [ex72_instance()] if graph == "ex72" else Lr.eval_instances(F)
    batch = ts.Batch(insts)
    runs = []
    k = K_OSC if objective == "oscillation" else 10
    # Oscillation attackers are also run with their exploration noise: a randomised attacker is still an attacker.
    modes = [(d_, False) for d_ in (+1, -1)] + ([(+1, True), (-1, True)] if objective == "oscillation" else [])
    for direc, sample in modes:
        torch.manual_seed(0)
        att = Lr.LearnedAttack(pol, T=T, k=k, direc=direc, sample=sample)
        r = ts.run(batch, ts.BASELINES[method](), T, ts.power_alpha(), att)
        ex = ts.range_excess(batch, ts.BASELINES[method](), Lr.LearnedAttack(pol, T=T, k=k, direc=direc, sample=sample))
        runs.append({k_: r[k_].numpy() for k_ in ("error", "spread", "hull_violation", "oscillation")} | {"excess": ex.numpy()})
    key = "error" if objective == "bias" else "oscillation"
    pick = np.argmax([r_[key] for r_ in runs], 0)
    best = {k_: np.choose(pick, [r_[k_] for r_ in runs]) for k_ in runs[0]}
    osc_det = float(np.maximum(runs[0]["oscillation"], runs[1]["oscillation"]).mean())
    return {"oscillation_deterministic": osc_det,"per_instance": {k: v.tolist() for k, v in best.items()},
            "error": float(best["error"].mean()), "oscillation": float(best["oscillation"].mean()),
            "spread": float(best["spread"].mean()), "hull_violation": float(best["hull_violation"].max()),
            "excess": float(best["excess"].max()),
            "error_dir+": float(runs[0]["error"].mean()), "error_dir-": float(runs[1]["error"].mean())}


def scripted_reference():
    """Phase 3 numbers on the same instances: best single scripted attack, and
    the per-instance best of all scripted attacks."""
    raw = json.loads((ROOT / "results" / "phase3" / "raw.json").read_text())["grid"]
    ref = {}
    for F in (1, 2):
        for m in F1:
            rows = [r for r in raw if r["F"] == F and r["method"] == m and r["attack"] != "none"]
            attacks = sorted({r["attack"] for r in rows})
            per = {a: np.array([r["error"] for r in sorted(rows, key=lambda r: r["seed"]) if r["attack"] == a]) for a in attacks}
            best_single = max(attacks, key=lambda a: per[a].mean())
            osc = {a: np.mean([r["oscillation"] for r in rows if r["attack"] == a]) for a in attacks}
            ref[(m, F)] = {"best_attack": best_single, "best_error": float(per[best_single].mean()),
                           "per_instance_best": float(np.max(np.stack(list(per.values())), 0).mean()),
                           "switching_osc": float(osc["switching"]),
                           "none": float(np.mean([r["error"] for r in raw if r["F"] == F and r["method"] == m and r["attack"] == "none"]))}
    return ref


def ex72_switching_reference():
    """The hand-built switching attack of [1, Ex. 7.2] against LF, on the same horizon."""
    I = ex72_instance()
    att = A.Switching(low=[0, 1, 2], odd=3)
    r = d.simulate(I["adj"], LocalFunctions.quadratic(I["a"], L=100), I["x0"], d.LF(1), T, d.power_steps(0.5, 0.6),
                   adversaries=I["adv"], attack=att)
    return {"LF": r.oscillation(T // 2)}


def report(results):
    ref = scripted_reference()
    lines = ["# Phase 4: RL attacker", "",
             f"PPO, {results[0]['iters']} iterations per attacker; evaluated on the Phase 3 benchmark instances "
             "(T = 2000, 10 graphs per F). The attacker picks its push direction per instance. "
             "Hull violation is over t ≥ T/2; excess is the α = 0 safety run.", "",
             "## Bias attacker: mean error |x̄(T) − x*|", "",
             "| Method | F | No attack | Best scripted attack (Phase 3) | Per-instance best of all scripted | **RL attacker** | RL / best scripted | Max hull violation | Max excess (α = 0) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r["objective"] != "bias":
            continue
        s = ref[(r["method"], r["F"])]
        lines.append(f"| {r['method']} | {r['F']} | {s['none']:.3f} | {s['best_error']:.3f} ({s['best_attack']}) | "
                     f"{s['per_instance_best']:.3f} | **{r['error']:.3f}** | {r['error'] / s['best_error']:.2f}× | "
                     f"{r['hull_violation']:.3f} | {r['excess']:.3f} |")
    lines += ["", "## Oscillation attacker: total variation of x̄ over t ≥ T/2", "",
              "| Defence | Graph | Scripted switching | **RL attacker** (best of deterministic / sampled) | RL attacker, deterministic only | Max hull violation |", "|---|---|---|---|---|---|"]
    ex72 = ex72_switching_reference()
    for r in results:
        if r["objective"] != "oscillation":
            continue
        refv = ex72[r["method"]] if r["graph"] == "ex72" else ref[(r["method"], r["F"])]["switching_osc"]
        name = "Ex. 7.2 (K5)" if r["graph"] == "ex72" else "random, F = 1"
        lines.append(f"| {r['method']} | {name} | {refv:.3f} | **{r['oscillation']:.3f}** | {r['oscillation_deterministic']:.3f} | {r['hull_violation']:.3f} |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


COLORS = {"DGD": "#5b5b5b", "LF": "#2a78d6", "LF-Metropolis": "#1baf7a", "Su-Vaidya": "#eda100",
          "Q-consensus": "#b3477a", "Q-consensus (fixed beta)": "#e87ba4", "DRC-RSSG": "#eb6834"}


def curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, F in zip(axes, (1, 2)):
        for r in (r for r in results if r["objective"] == "bias" and r["F"] == F):
            y = np.convolve(r["curve"], np.ones(10) / 10, mode="valid")
            ax.plot(np.maximum(y, 1e-2), color=COLORS[r["method"]], lw=1.4, label=r["method"])
        ax.set(xlabel="PPO iteration", ylabel="push dir·(x̄(T) − x*) on training graphs\n(10-iteration moving average, log)",
               title=f"F = {F}", yscale="log")
        ax.grid(axis="y", color="#e6e5df", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    axes[1].set_ylabel("")
    fig.subplots_adjust(wspace=0.18)
    fig.suptitle("Phase 4: RL attacker learning curves", fontsize=11)
    fig.savefig(OUT / "curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(m, 1, "bias", "random", args.iters, 0) for m in F1]
    jobs += [(m, 2, "bias", "random", args.iters, 0) for m in F2]
    jobs += [("LF", 1, "oscillation", "random", args.iters, 0), ("LF", 1, "oscillation", "ex72", args.iters, 0)]
    with get_context("spawn").Pool(args.procs) as pool:
        results = pool.map(train_job, jobs, chunksize=1)
    for r in results:
        r["iters"] = args.iters
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    curves(results)
    print(report(results))
