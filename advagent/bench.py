"""Shared benchmark: the scripted attacks in the batched simulator.

``evaluate`` runs one defence against one attack on a list of instances and
returns per-instance metrics with the same definitions as
``experiments/01_scripted_attacks.py``, which ``tests/test_torchsim.py`` ties to the NumPy
simulator.
"""

import numpy as np
import torch

from . import graphs as g
from . import torchsim as ts

T = 2000
UNSAFE_TOL, SPREAD_TOL = 1e-3, 1.0

# name -> factory(F) returning a ts.Attack, None (no adversaries) or ("stealthy", fake)
SCRIPTED = {
    "none": lambda F: None,
    "fixed +50": lambda F: ts.Fixed(50.0),
    "random Byzantine": lambda F: ts.UniformRandom(-50, 50),
    "farthest evader": lambda F: ts.FarthestEvader(+1, 0.9),
    "stealthy (fake 5)": lambda F: ("stealthy", 5.0),
    "stealthy (fake 50)": lambda F: ("stealthy", 50.0),
    "trim-hugger": lambda F: ts.Band(0.99),
    "switching": lambda F: ts.StallSwitcher(),
}


def prepare(insts, attack):
    """Instances and attack object for the simulator."""
    if attack is None:
        return [dict(I, adv=[]) for I in insts], None
    if isinstance(attack, tuple) and attack[0] == "stealthy":
        return ts.stealthy(insts, attack[1]), None
    return insts, attack


@torch.no_grad()
def evaluate(rule_factory, insts, attack_factory, T=T, seed=0, device="cpu"):
    """attack_factory() -> attack spec (see SCRIPTED). Fresh rule/attack per run."""
    torch.manual_seed(seed)
    run_insts, att = prepare(insts, attack_factory())
    batch = ts.Batch(run_insts, device=device)
    r = ts.run(batch, rule_factory(), T, ts.power_alpha(), att)
    _, att0 = prepare(insts, attack_factory())
    ex = ts.range_excess(batch, rule_factory(), att0)
    out = {k: r[k].cpu().numpy() for k in ("error", "spread", "hull_violation", "oscillation")}
    out["excess"] = ex.cpu().numpy()
    return out


def flag(m):
    """'unsafe' / 'no consensus' / '' for a dict of per-instance metrics."""
    if (m["hull_violation"] > UNSAFE_TOL).any() or (m["excess"] > UNSAFE_TOL).any():
        return "unsafe"
    if (m["spread"] > SPREAD_TOL).any():
        return "no consensus"
    return ""


MARK = {"unsafe": " ⚠", "no consensus": " ∅", "": ""}


def uw_instance(K=5, b=4.0):
    """Ex. 8.2 of [1]: U-W graph, f = x^2 on W and (x - b)^2 on U, no adversaries."""
    adj, W, U = g.uw_graph(K)
    a = np.zeros(adj.shape[0])
    a[U] = b
    return {"adj": adj, "adv": [], "a": a, "x0": a.copy(), "F": 1}
