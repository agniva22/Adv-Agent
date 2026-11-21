"""Figures of the numerical section (Figs. 1-3 of the paper).

Fig. 1  An example network (F = 1) and the states of its regular agents over time:
        DGD under a fixed adversary (captured), LF and memory + fallback, each under
        its own best-response RL attacker. The instance is the unseen graph with the
        median improvement of memory + fallback over LF (no cherry-picking).
Fig. 2  Error |x_bar(t) - x*| over time, averaged over the 100 unseen graphs, for the
        safe methods under their own best-response RL attacker (F = 1 and F = 2).
Fig. 3  Per-graph error under the best-response attacker: LF vs memory + fallback.

Stage 1 (simulation) writes results/paper_figures/data.npz; stage 2
(``experiments/plot_figures.py``) draws the panels into results/paper_figures/.
    python experiments/paper_figures.py [--plot-only]
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("ADV_P7_UNSEEN", "1")
from advagent import learn as Lr
from advagent import torchsim as ts

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "paper_figures" / "data.npz"
P7 = ROOT / "results" / "phase7_unseen" / "results.json"
BR = "RL attacker (best response)"
T = 2000
# Fixed order (entity -> colour never changes); our methods first. Reference categorical palette, slots 1-6.
METHODS = ["memory + fallback", "memory + fallback (game)", "LF", "LF-Metropolis", "Trimmed mean", "Su-Vaidya"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STYLES = ["-", "-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]


def plot_figures():
    spec = importlib.util.spec_from_file_location("plot_figures", ROOT / "experiments" / "plot_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def phase7():
    spec = importlib.util.spec_from_file_location("phase7_eval", ROOT / "experiments" / "phase7_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rule_of(p7, name):
    return ts.BASELINES[name]() if name in ts.BASELINES else p7.learned(name)


# ------------------------------------------------------------------ stage 1

@torch.no_grad()
def error_curves(p7, name, F, insts):
    """Mean over graphs of |x_bar(t) - x*| under the method's best-response attacker,
    taking the worse push direction per graph (as in the evaluation)."""
    rule, pol = rule_of(p7, name), p7.br_attacker(name, F)
    curves, finals = [], []
    for i in range(0, len(insts), 10):
        batch = ts.Batch(insts[i:i + 10])
        runs = []
        for dr in (1, -1):
            torch.manual_seed(0)
            r = ts.run(batch, rule, T, ts.power_alpha(), Lr.LearnedAttack(pol, direc=dr))
            runs.append((r["mean"] - batch.x_star[None]).abs())          # (T+1, B)
        pick = runs[0][-1] >= runs[1][-1]
        e = torch.where(pick[None], runs[0], runs[1])
        curves.append(e.numpy())
        finals.append(e[-1].numpy())
    print(f"  curves: {name}, F={F}", flush=True)
    return np.concatenate(curves, 1).mean(1), np.concatenate(finals)


@torch.no_grad()
def trajectory(rule, inst, attack):
    """All states over time for one instance (B = 1)."""
    batch = ts.Batch([inst])
    rule.reset(batch)
    astate = attack.init(batch) if attack is not None else ()
    x, alpha, xs = batch.x0.clone(), ts.power_alpha(), [batch.x0[0].numpy().copy()]
    for t in range(T):
        x, astate = ts.step(t, x, batch, rule, alpha, attack, astate)
        xs.append(x[0].numpy().copy())
    return np.array(xs), float(batch.x_star[0])


def worse_trajectory(p7, name, inst, F):
    pol = p7.br_attacker(name, F)
    best = None
    for dr in (1, -1):
        torch.manual_seed(0)
        xs, xstar = trajectory(rule_of(p7, name), inst, Lr.LearnedAttack(pol, direc=dr))
        reg = np.ones(xs.shape[1], bool)
        reg[list(inst["adv"])] = False
        err = abs(xs[-1, reg].mean() - xstar)
        if best is None or err > best[0]:
            best = (err, xs)
    return best[1]


def simulate():
    p7 = phase7()
    torch.set_num_threads(4)
    out = {}
    for F in (1, 2):
        insts = Lr.train_pool(F, 100, offset=300_000)   # the Phase 7 unseen graphs
        for name in METHODS:
            curve, finals = error_curves(p7, name, F, insts)
            out[f"curve|{name}|F{F}"] = curve
            out[f"final|{name}|F{F}"] = finals
    # Example: the unseen F = 1 graph with the median improvement of memory + fallback over LF.
    grid = json.loads(P7.read_text())["grid"]
    gain = np.array(grid[f"LF|F1|{BR}"]["error"]) - np.array(grid[f"memory + fallback|F1|{BR}"]["error"])
    k = int(np.argsort(gain)[len(gain) // 2])
    inst = Lr.train_pool(1, 100, offset=300_000)[k]
    torch.manual_seed(0)
    out["ex|DGD"], xstar = trajectory(ts.BASELINES["DGD"](), inst, ts.Fixed(50.0))
    out["ex|LF"] = worse_trajectory(p7, "LF", inst, 1)
    out["ex|memory + fallback"] = worse_trajectory(p7, "memory + fallback", inst, 1)
    reg = np.ones(inst["adj"].shape[0], bool)
    reg[list(inst["adv"])] = False
    out.update({"ex_adj": inst["adj"], "ex_reg": reg, "ex_a": inst["a"], "ex_xstar": np.array(xstar), "ex_index": np.array(k)})
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA, **out)
    for F in (1, 2):   # consistency with Table II (same graphs, same attackers)
        print(f"  F={F} final BR errors:", {n: round(float(out[f'final|{n}|F{F}'].mean()), 3) for n in METHODS})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        simulate()
    plot_figures().main()
