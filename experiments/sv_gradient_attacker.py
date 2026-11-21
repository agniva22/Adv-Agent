"""An RL attacker against Su-Vaidya [5] that also learns its fake gradients.

The Phase 4 attacker against Su-Vaidya controls only the values it sends; its
fake gradients follow the fixed rule ``torchsim.gradient_lie``. Here
``learn.GradientAttacker`` adds a second per-receiver action that places the
fake gradient, warm-started from the Phase 4 attacker (so it starts exactly
where that one is) and trained 300 more PPO iterations per F. Both attackers
are then evaluated against Su-Vaidya on the benchmark and on 100 unseen
graphs (worse push direction per graph). The new attacker is saved next to the
Phase 4 ones as ``Su-Vaidya_F{F}_bias_random_grad.pt``; Phase 7 uses it.
Writes results/sv_gradient_attacker/{table.md, results.json}.
    PYTHONPATH=. python experiments/sv_gradient_attacker.py [--iters N]
"""

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from advagent import bench
from advagent import learn as Lr
from advagent import torchsim as ts

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "results" / "phase4" / "attackers"
OUT = ROOT / "results" / "sv_gradient_attacker"


def old(F):
    pol = Lr.AttackerPolicy()
    pol.load_state_dict(torch.load(P4 / f"Su-Vaidya_F{F}_bias_random.pt", map_location="cpu"))
    return pol.eval()


def train_job(job):
    F, iters = job
    torch.set_num_threads(5)
    log = lambda s: open(OUT / f"train_F{F}.log", "a").write(s + "\n")
    pol, curve = Lr.train_attacker(ts.BASELINES["Su-Vaidya"], Lr.train_pool(F, 512), iters=iters,
                                   policy=Lr.GradientAttacker.from_policy(old(F)), seed=3, log=log)
    torch.save(pol.state_dict(), P4 / f"Su-Vaidya_F{F}_bias_random_grad.pt")
    return F, curve


@torch.no_grad()
def damage(pol, insts):
    runs = [bench.evaluate(lambda: ts.BASELINES["Su-Vaidya"](), insts, lambda dr=dr: Lr.LearnedAttack(pol, direc=dr))
            for dr in (1, -1)]
    worst = np.maximum(runs[0]["error"], runs[1]["error"])
    unsafe = any(bench.flag(r) == "unsafe" for r in runs)
    return float(worst.mean()), unsafe


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with get_context("spawn").Pool(2) as pool:
        curves = dict(pool.map(train_job, [(F, args.iters) for F in (1, 2)]))
    torch.set_num_threads(10)
    lines = ["# Su-Vaidya against an attacker that also learns its fake gradients", "",
             "Mean error of Su-Vaidya [5] under each attacker (worse push direction per graph).", "",
             "| F | Graphs | Values learned, fixed gradient lie (Phase 4) | Values and gradients learned | Unsafe? |",
             "|---|---|---|---|---|"]
    res = {}
    for F in (1, 2):
        new = Lr.GradientAttacker()
        new.load_state_dict(torch.load(P4 / f"Su-Vaidya_F{F}_bias_random_grad.pt", map_location="cpu"))
        new.eval()
        for split, insts in (("benchmark (10)", Lr.eval_instances(F)), ("unseen (100)", Lr.train_pool(F, 100, offset=300_000))):
            (e0, u0), (e1, u1) = damage(old(F), insts), damage(new, insts)
            res[f"F{F}|{split}"] = {"phase4": e0, "gradient": e1, "unsafe": u0 or u1}
            lines.append(f"| {F} | {split} | {e0:.3f} | **{e1:.3f}** | {'yes' if u0 or u1 else 'no'} |")
    (OUT / "results.json").write_text(json.dumps({"results": res, "curves": curves}))
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
