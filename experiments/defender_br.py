"""Phase 7 protocol for the memory + fallback defender: does its gain survive an attacker trained against it?

Best-response attackers (PPO, warm start from the Phase 4 LF attacker, 200 more iterations,
as in Phase 7) are trained against the memory + fallback defender and, for a fair budget,
again against LF. Two scripted attacks target the memory: a sleeper (honest, then
trim-hugging from t = 500 or 1000) and a flicker (honest and trim-hugging in turns of 25
steps, a little longer than the memory's ~20-step horizon).
Writes results/defender_br/{br.md, br.json, curves.json, attackers/*.pt}.
    PYTHONPATH=. python experiments/defender_br.py [--iters N] [--eval-only]
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
OUT = ROOT / "results" / "defender_br"
P4 = ROOT / "results" / "phase4" / "attackers"
DEF = "memory + fallback"
BR = "RL attacker (best response)"
UNSEEN = 100


class Sleeper(ts.Attack):
    """Honest (band action 0: send the receiver its own value) until ``start``,
    then trim-hugging; with ``period``, alternate the two every ``period`` steps."""

    def __init__(self, start=0, period=None, a=0.99, name="sleeper"):
        self.start, self.period, self.a, self.name = start, period, a, name

    def __call__(self, t, x, vals, batch, state):
        on = (t // self.period) % 2 == 1 if self.period else t >= self.start
        v, _ = ts.band(torch.full_like(x, self.a if on else 0.0), x, vals, batch)
        return v[..., None].expand_as(vals), state


ATTACKS = dict(bench.SCRIPTED)
ATTACKS.update({
    "sleeper (t = 500)": lambda F: Sleeper(500),
    "sleeper (t = 1000)": lambda F: Sleeper(1000),
    "flicker (25 steps)": lambda F: Sleeper(period=25),
})


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def defender(name, device="cpu"):
    if name == DEF:
        return module("defender_trust").load(DEF).to(device)
    return ts.BASELINES[name]()


def fname(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


def br_job(job):
    name, F, iters, device = job
    torch.set_num_threads(5)
    rule = defender(name, device)
    pol = Lr.AttackerPolicy()
    pol.load_state_dict(torch.load(P4 / f"LF_F{F}_bias_random.pt", map_location="cpu"))   # warm start, as in Phase 7
    log_path = OUT / f"train_{fname(name)}_F{F}.log"
    log = lambda s: open(log_path, "a").write(s + "\n")
    pol, curve = Lr.train_attacker(lambda: rule, Lr.train_pool(F, 512), iters=iters, policy=pol, seed=7,
                                   device=device, log=log)
    (OUT / "attackers").mkdir(parents=True, exist_ok=True)
    torch.save(pol.state_dict(), OUT / "attackers" / f"{fname(name)}_F{F}.pt")
    return {"name": name, "F": F, "curve": curve}


def attacker(path):
    pol = Lr.AttackerPolicy()
    pol.load_state_dict(torch.load(path, map_location="cpu"))
    return pol.eval()


def learned_attacks(name, F):
    """RL attackers to run against ``name``: its own best response, and the Phase 4 one against LF."""
    out = {"RL attacker (vs LF, Phase 4)": attacker(P4 / f"LF_F{F}_bias_random.pt")}
    own = OUT / "attackers" / f"{fname(name)}_F{F}.pt"
    if own.exists():
        out[BR] = attacker(own)
    elif (P4 / f"{name.replace(' ', '_')}_F{F}_bias_random.pt").exists():
        out[BR] = attacker(P4 / f"{name.replace(' ', '_')}_F{F}_bias_random.pt")   # Phase 4 best response
    return out


def eval_on(methods, insts_for, device):
    res = {}
    for F in (1, 2):
        insts = insts_for(F)
        for name in methods:
            fac = lambda name=name: defender(name, device)
            with torch.no_grad():
                for att, af in ATTACKS.items():
                    res[(name, F, att)] = bench.evaluate(fac, insts, lambda af=af, F=F: af(F), device=device)
                for att, pol in learned_attacks(name, F).items():
                    runs = [bench.evaluate(fac, insts, lambda dr=dr: Lr.LearnedAttack(pol.to(device), direc=dr), device=device)
                            for dr in (1, -1)]
                    pick = np.argmax([runs[0]["error"], runs[1]["error"]], 0)
                    res[(name, F, att)] = {k: np.where(pick == 0, runs[0][k], runs[1][k]) for k in runs[0]}
            print(f"  evaluated {name} F={F} on {len(insts)} graphs", flush=True)
    return res


def table(res, names, title, n_graphs):
    lines = [f"## {title}", "",
             f"Mean error over {n_graphs} graphs per F (T = 2000). '{BR}' is trained against that method "
             "(memory + fallback, LF: here, 200 PPO iterations from the Phase 4 LF attacker; Su–Vaidya: Phase 4). "
             "Mean and worst are over all attacks except 'none'.", ""]
    for F in (1, 2):
        atts = [a for a in list(ATTACKS) + ["RL attacker (vs LF, Phase 4)", BR] if all((n, F, a) in res for n in names)]
        lines += [f"### F = {F}", "", "| Method | No attack | Mean over attacks | Worst attack | Own best-response attacker | Safe? |",
                  "|---|---|---|---|---|---|"]
        for n in names:
            errs = {a: res[(n, F, a)]["error"].mean() for a in atts if a != "none"}
            worst = max(errs, key=errs.get)
            flags = {bench.flag(res[(n, F, a)]) for a in atts} - {""}
            lines.append(f"| {n} | {res[(n, F, 'none')]['error'].mean():.3f} | {np.mean(list(errs.values())):.3f} | "
                         f"{errs[worst]:.3f} ({worst}) | {errs[BR]:.3f} | {'yes' if not flags else ', '.join(sorted(flags))} |")
        lines += ["", "| Attack | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for a in atts:
            lines.append(f"| {a} | " + " | ".join(f"{res[(n, F, a)]['error'].mean():.3f}" for n in names) + " |")
        lines.append("")
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        jobs = [(n, F, args.iters, args.device) for n in (DEF, "LF") for F in (1, 2)]
        with get_context("spawn").Pool(len(jobs)) as pool:
            curves = pool.map(br_job, jobs, chunksize=1)
        (OUT / "curves.json").write_text(json.dumps(curves))
    names = ["LF", "Su-Vaidya", DEF]
    out, md = {}, ["# Memory + fallback defender against its own best-response attacker", ""]
    for key, insts_for, title in (("unseen", lambda F: Lr.train_pool(F, UNSEEN, offset=300_000), f"{UNSEEN} unseen graphs"),
                                  ("benchmark", Lr.eval_instances, "Phase 3 benchmark (10 graphs)")):
        res = eval_on(names, insts_for, args.device)
        out[key] = {f"{n}|F{F}|{a}": {k: v.tolist() for k, v in m.items()} for (n, F, a), m in res.items()}
        md += table(res, names, title, UNSEEN if key == "unseen" else 10)
    (OUT / "br.json").write_text(json.dumps(out))
    (OUT / "br.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
