"""Larger networks (n = 50 and 100); every policy was trained on 20-node graphs only.

Instances are drawn like ``learn.instance`` with n = 50 and 100 nodes and the same
share of adversaries (up to 3F per 20 nodes). Methods: LF, LF-Metropolis, Su-Vaidya,
Trimmed mean and the two memory + fallback defenders, unchanged. Attacks: the scripted
attacks and each method's own best-response RL attacker (trained on 20 nodes; all
policies act per node, so they apply as they are).
Writes results/larger_networks/{table.md, results.json}.

    python experiments/09_larger_networks.py [--sizes 50 100] [--graphs N] [--workers W]
"""

import argparse
import json
from multiprocessing import get_context

import numpy as np
import torch

from advagent import bench
from advagent import checkpoints as C
from advagent import graphs as g
from advagent import learn as Lr
from advagent import torchsim as ts

OUT = C.LARGER_NETWORKS
BR = "RL attacker (best response)"
NAMES = ["LF", "LF-Metropolis", "Su-Vaidya", "Trimmed mean", "memory + fallback", "memory + fallback (game)"]
CHUNK = 10   # graphs per simulator batch: the memory defenders keep per-step memory (RAM)


def instance(seed, F, n):
    """``learn.instance`` on n nodes, with up to 3F adversaries per 20 nodes."""
    rng = np.random.default_rng(seed)
    adj = g.random_robust(n, 2 * F + 1, rng, extra=1)
    adv = g.random_f_local_set(adj, F, 3 * F * n // 20, rng)
    return {"adj": adj, "adv": adv, "a": rng.uniform(-5, 5, n), "x0": rng.uniform(-5, 5, n), "F": F}


def evaluate(fac, insts, af):
    parts = [bench.evaluate(fac, insts[i:i + CHUNK], af) for i in range(0, len(insts), CHUNK)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def eval_job(job):
    name, F, n, graphs = job
    torch.set_num_threads(4)
    insts = [instance(400_000 + 10_000 * n + 1000 * F + s, F, n) for s in range(graphs)]
    r = ts.BASELINES[name]() if name in ts.BASELINES else C.defender(name)
    res = {}
    with torch.no_grad():
        for att, af in bench.SCRIPTED.items():
            res[att] = evaluate(lambda: r, insts, lambda af=af: af(F))
        pol = C.best_response_attacker(name, F)
        runs = [evaluate(lambda: r, insts, lambda dr=dr: Lr.LearnedAttack(pol, direc=dr)) for dr in (1, -1)]
    pick = np.argmax([runs[0]["error"], runs[1]["error"]], 0)
    res[BR] = {k: np.where(pick == 0, runs[0][k], runs[1][k]) for k in runs[0]}
    print(f"  evaluated {name} F={F} n={n}", flush=True)
    return name, F, n, {a: {k: v.tolist() for k, v in m.items()} for a, m in res.items()}


def table(res, sizes, graphs):
    lines = ["# Bigger graphs", "",
             f"Mean error over {graphs} random graphs per (n, F), T = 2000. All policies were trained on 20 nodes only. "
             f"Mean and worst are over all attacks except 'none', including '{BR}' (the method's own).", ""]
    for n in sizes:
        for F in (1, 2):
            lines += [f"## n = {n}, F = {F}", "",
                      "| Method | No attack | Mean over attacks | Worst attack | Own best response | Safe? |", "|---|---|---|---|---|---|"]
            for name in NAMES:
                m = res[(name, F, n)]
                att = {a: float(np.mean(v["error"])) for a, v in m.items() if a != "none"}
                worst = max(att, key=att.get)
                flags = {bench.flag({k: np.array(x) for k, x in v.items()}) for v in m.values()} - {""}
                lines.append(f"| {name} | {np.mean(m['none']['error']):.3f} | {np.mean(list(att.values())):.3f} | "
                             f"{att[worst]:.3f} ({worst}) | {att[BR]:.3f} | {'yes' if not flags else ', '.join(sorted(flags))} |")
            lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--graphs", type=int, default=50)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(name, F, n, args.graphs) for n in args.sizes for F in (1, 2) for name in NAMES]
    with get_context("spawn").Pool(args.workers) as pool:
        out = pool.map(eval_job, jobs, chunksize=1)
    res = {(name, F, n): r for name, F, n, r in out}
    (OUT / "results.json").write_text(json.dumps({f"{name}|F{F}|n{n}": r for (name, F, n), r in res.items()}))
    md = table(res, args.sizes, args.graphs)
    (OUT / "table.md").write_text(md)
    print(md)
