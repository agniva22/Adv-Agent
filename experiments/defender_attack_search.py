"""A stronger attacker search against the memory + fallback defender.

The best-response attacker of ``defender_br.py`` did not improve on its warm
start, so here we search harder, with three kinds of attacker per F:

  from scratch   fresh policy (no warm start), 400 PPO iterations, bias objective;
  swing          fresh policy, oscillation objective (Phase 4 settings: a decision
                 every 50 steps, low exploration noise, lr 1e-3), 300 iterations;
  memory-aware   sees the receiver's memory of the adversary slots
                 (``learn.MemoryAwareAttacker``), warm start from the Phase 4 LF
                 attacker, 300 iterations (only against the defender: LF has no memory).

From scratch and swing are trained against LF too, for a fair budget. Every method
then faces every learned attacker (these, plus the earlier best responses and the
Phase 4 LF attacker), each run with both push directions (and sampled actions for
swing); we keep the worst per graph.
Writes results/defender_attack_search/{search.md, search.json, curves.json, attackers/*.pt}.
    PYTHONPATH=. python experiments/defender_attack_search.py [--eval-only]
"""

import argparse
import importlib.util
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from advagent import bench
from advagent import learn as Lr
from advagent import torchsim as ts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "defender_attack_search"
P4 = ROOT / "results" / "phase4" / "attackers"
BRD = ROOT / "results" / "defender_br" / "attackers"
DEF = "memory + fallback"
METHODS = ["LF", "Su-Vaidya", DEF]
KINDS = {   # kind: (objective, k, iters, targets)
    "from scratch": ("bias", 10, 400, ("LF", DEF)),
    "swing": ("oscillation", 50, 300, ("LF", DEF)),
    "memory-aware": ("bias", 10, 300, (DEF,)),
}
UNSEEN = 100


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def defender(name):
    if name == DEF:
        return module("defender_trust").load(DEF)
    return ts.BASELINES[name]()


def fname(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


def new_policy(kind, F):
    if kind == "memory-aware":
        base = Lr.AttackerPolicy()
        base.load_state_dict(torch.load(P4 / f"LF_F{F}_bias_random.pt", map_location="cpu"))
        return Lr.MemoryAwareAttacker.from_policy(base), 3e-4
    pol = Lr.AttackerPolicy()
    if kind == "swing":
        # Low exploration noise: otherwise noise alone earns the reward (Phase 4).
        with torch.no_grad():
            pol.log_std.fill_(-2.0)
        return pol, 1e-3
    return pol, 3e-4


def train_job(job):
    kind, target, F = job
    torch.set_num_threads(2)
    objective, k, iters, _ = KINDS[kind]
    rule = defender(target)
    pol, lr = new_policy(kind, F)
    if kind == "memory-aware":
        pol.watch(rule)
    log_path = OUT / f"train_{fname(kind)}_vs_{fname(target)}_F{F}.log"
    log = lambda s: open(log_path, "a").write(s + "\n")
    pol, curve = Lr.train_attacker(lambda: rule, Lr.train_pool(F, 512), k=k, iters=iters, policy=pol, lr=lr,
                                   seed=11, objective=objective, log=log)
    (OUT / "attackers").mkdir(parents=True, exist_ok=True)
    torch.save(pol.state_dict(), OUT / "attackers" / f"{fname(kind)}_vs_{fname(target)}_F{F}.pt")
    return {"kind": kind, "target": target, "F": F, "curve": curve}


def attackers(F):
    """{label: (policy, k, sampled modes too?)} for every learned attacker at this F."""
    out = {}

    def add(label, path, cls=Lr.AttackerPolicy, k=10, sample=False):
        if path.exists():
            pol = cls()
            pol.load_state_dict(torch.load(path, map_location="cpu"))
            out[label] = (pol.eval(), k, sample)

    add("Phase 4 (vs LF)", P4 / f"LF_F{F}_bias_random.pt")
    for t in ("LF", DEF):
        add(f"best response, warm start (vs {t})", BRD / f"{fname(t)}_F{F}.pt")
    for kind, (objective, k, _, targets) in KINDS.items():
        for t in targets:
            add(f"{kind} (vs {t})", OUT / "attackers" / f"{fname(kind)}_vs_{fname(t)}_F{F}.pt",
                cls=Lr.MemoryAwareAttacker if kind == "memory-aware" else Lr.AttackerPolicy, k=k, sample=kind == "swing")
    return out


def eval_job(job):
    """Every learned attacker against one method at one F, on one instance set."""
    name, F, split = job
    torch.set_num_threads(4)
    insts = Lr.train_pool(F, UNSEEN, offset=300_000) if split == "unseen" else Lr.eval_instances(F)
    rule = defender(name)
    res = {}
    for label, (pol, k, sample) in attackers(F).items():
        if isinstance(pol, Lr.MemoryAwareAttacker):
            pol.watch(rule)
        runs = []
        for direc, smp in [(1, False), (-1, False)] + ([(1, True), (-1, True)] if sample else []):
            with torch.no_grad():
                runs.append(bench.evaluate(lambda: rule, insts, lambda: Lr.LearnedAttack(pol, k=k, direc=direc, sample=smp)))
        pick = np.argmax([r["error"] for r in runs], 0)
        res[label] = {key: np.choose(pick, [r[key] for r in runs]).tolist() for key in runs[0]}
        res[label]["max_oscillation"] = np.max([r["oscillation"] for r in runs], 0).tolist()
    return name, F, split, res


def table(results, split, n):
    lines = [f"## {split} ({n} graphs per F)", "",
             "Mean error per learned attacker (worst push direction per graph; swing also with sampled actions). "
             "'Worst' is the worst attacker for that method, by mean error.", ""]
    for F in (1, 2):
        labels = list(results[(METHODS[0], F, split)])
        lines += [f"### F = {F}", "", "| Attacker | " + " | ".join(METHODS) + " |", "|---" * (len(METHODS) + 1) + "|"]
        for lab in labels:
            lines.append(f"| {lab} | " + " | ".join(f"{np.mean(results[(m, F, split)][lab]['error']):.3f}" for m in METHODS) + " |")
        worst = {m: max(labels, key=lambda l: np.mean(results[(m, F, split)][l]["error"])) for m in METHODS}
        lines.append("| **worst** | " + " | ".join(f"**{np.mean(results[(m, F, split)][worst[m]]['error']):.3f}** ({worst[m]})"
                                                for m in METHODS) + " |")
        osc = [lab for lab in labels if lab.startswith("swing")]
        if osc:
            lines.append("| swing: oscillation of x̄ (t ≥ T/2), worst | " + " | ".join(
                f"{max(np.mean(results[(m, F, split)][l]['max_oscillation']) for l in osc):.3f}" for m in METHODS) + " |")
        unsafe = {m: [l for l in labels if bench.flag({k: np.array(v) for k, v in results[(m, F, split)][l].items()}) == "unsafe"]
                  for m in METHODS}
        lines += ["", "Safe? " + "; ".join(f"{m}: {'yes' if not u else 'NO (' + ', '.join(u) + ')'}" for m, u in unsafe.items()), ""]
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        jobs = [(kind, t, F) for kind, (_, _, _, targets) in KINDS.items() for t in targets for F in (1, 2)]
        with get_context("spawn").Pool(len(jobs)) as pool:
            curves = pool.map(train_job, jobs, chunksize=1)
        (OUT / "curves.json").write_text(json.dumps(curves))
    jobs = [(m, F, s) for s in ("unseen", "benchmark") for m in METHODS for F in (1, 2)]
    with get_context("spawn").Pool(6) as pool:
        out = pool.map(eval_job, jobs, chunksize=1)
    results = {(m, F, s): r for m, F, s, r in out}
    (OUT / "search.json").write_text(json.dumps({f"{m}|F{F}|{s}": r for (m, F, s), r in results.items()}))
    md = ["# Stronger attacker search against the memory + fallback defender", ""]
    md += table(results, "unseen", UNSEEN) + table(results, "benchmark", 10)
    (OUT / "search.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
