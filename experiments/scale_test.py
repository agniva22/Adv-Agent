"""Do the results hold on bigger graphs? Every policy was trained on 20-node graphs only.

Instances are drawn like ``learn.instance`` with n = 50 and 100 nodes, and the same
share of adversaries (up to 3F per 20 nodes). Methods: LF, LF-Metropolis, Su-Vaidya,
Trimmed mean and the memory + fallback defenders of Phase 7, unchanged. Attacks: the
scripted attacks of Phase 3 and each method's own best-response RL attacker from
Phase 4 / Phase 7 (trained on 20 nodes; all policies act per node, so they apply as is).
Writes results/scale_test/{scale.md, scale.json}.
    PYTHONPATH=. python experiments/scale_test.py [--sizes 50 100] [--graphs N] [--workers W]
"""

import argparse
import importlib.util
import json
import os
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("ADV_P7_UNSEEN", "1")   # Phase 7 helpers: the unseen-graph best-response attackers
from advagent import bench
from advagent import graphs as g
from advagent import learn as Lr
from advagent import torchsim as ts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "scale_test"
BR = "RL attacker (best response)"
NAMES = ["LF", "LF-Metropolis", "Su-Vaidya", "Trimmed mean", "memory + fallback", "memory + fallback (game)"]
CHUNK = 10   # graphs per simulator batch: the memory defenders keep per-step memory (RAM)


def phase7():
    spec = importlib.util.spec_from_file_location("phase7_eval", ROOT / "experiments" / "phase7_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    p7 = phase7()
    insts = [instance(400_000 + 10_000 * n + 1000 * F + s, F, n) for s in range(graphs)]
    r = ts.BASELINES[name]() if name in ts.BASELINES else p7.learned(name)
    res = {}
    with torch.no_grad():
        for att, af in bench.SCRIPTED.items():
            res[att] = evaluate(lambda: r, insts, lambda af=af: af(F))
        pol = p7.br_attacker(name, F)
        runs = [evaluate(lambda: r, insts, lambda dr=dr: Lr.LearnedAttack(pol, direc=dr)) for dr in (1, -1)]
    pick = np.argmax([runs[0]["error"], runs[1]["error"]], 0)
    res[BR] = {k: np.where(pick == 0, runs[0][k], runs[1][k]) for k in runs[0]}
    print(f"  evaluated {name} F={F} n={n}", flush=True)
    return name, F, n, {a: {k: v.tolist() for k, v in m.items()} for a, m in res.items()}


def table(res, sizes, graphs):
    lines = ["# Bigger graphs", "",
             f"Mean error over {graphs} random graphs per (n, F), T = 2000. All policies were trained on 20 nodes only. "
             f"Mean and worst are over all attacks except 'none', including '{BR}' (the method's own, from Phase 4 / 7).", ""]
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
    (OUT / "scale.json").write_text(json.dumps({f"{name}|F{F}|n{n}": r for (name, F, n), r in res.items()}))
    md = table(res, args.sizes, args.graphs)
    (OUT / "scale.md").write_text(md)
    print(md)
