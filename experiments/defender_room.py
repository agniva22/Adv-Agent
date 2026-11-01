"""How much can any shielded defender gain over LF? (fix 0 for the defender)

An oracle gets its own free score for every (episode, node, slot) inside the
shield and optimises them directly on the benchmark instances, knowing x*. No
learned policy that sees only local information can do better against the same
fixed attack, so the oracle's error is a floor for the learned defender and
LF - oracle is the room there is to win.

Writes results/defender_fixes/room.{json,md}.
    PYTHONPATH=. python experiments/defender_room.py
"""

import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from advagent import bench
from advagent import learn as Lr
from advagent import torchsim as ts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "defender_fixes"
ATTACKS = {
    "none": lambda F: None,
    "fixed +50": lambda F: ts.Fixed(50.0),
    "trim-hugger": lambda F: ts.Band(0.99),
    "RL attacker (vs LF)": "rl",
}
CS = (0.25, 0.05)


class Oracle(torch.nn.Module, ts.Rule):
    """Floored softmax over self + LF-kept values with free per-slot scores."""

    name = "oracle"

    def __init__(self, B, n, d, c):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(B, n, d + 1))
        self.c = c

    def aggregate(self, x, vals, batch, t, at):
        keep = ts.lf_keep(x[..., None], vals, batch.nmask, batch.F)
        mask = torch.cat([torch.ones_like(keep[..., :1]), keep], -1)
        sm = torch.softmax(self.theta.masked_fill(~mask, -torch.inf), -1)
        eta = self.c / (batch.deg[..., None] + 1)
        w = torch.where(mask, eta + (1 - mask.sum(-1, keepdim=True) * eta) * sm, 0)
        return w[..., 0] * x + (w[..., 1:] * vals).sum(-1)


def attack_for(name, F):
    if ATTACKS[name] == "rl":
        pol = Lr.AttackerPolicy()
        pol.load_state_dict(torch.load(ROOT / "results" / "phase4" / "attackers" / f"LF_F{F}_bias_random.pt"))
        return Lr.LearnedAttack(pol.eval(), direc=1.0)
    return ATTACKS[name](F)


def job(args):
    F, att_name, c, iters = args
    torch.set_num_threads(4)
    torch.manual_seed(0)
    insts, att = bench.prepare(Lr.eval_instances(F), attack_for(att_name, F))
    batch = ts.Batch(insts)
    alpha = ts.power_alpha()
    with torch.no_grad():
        lf = ts.run(batch, ts.LF(), bench.T, alpha, att)["error"].numpy()
    orc = Oracle(batch.B, batch.n, batch.d, c)
    opt = torch.optim.Adam(orc.parameters(), lr=0.1)
    best = None
    for it in range(iters):
        r = ts.run(batch, orc, bench.T, alpha, att, ckpt=100)
        loss = Lr.defender_loss(r, batch).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        err = r["error"].detach().numpy()
        best = err if best is None else np.minimum(best, err)   # per instance, best weights seen
    with torch.no_grad():
        r = ts.run(batch, orc, bench.T, alpha, att)
        safe = float(r["hull_violation"].max())
    return {"F": F, "attack": att_name, "c": c, "lf": lf.tolist(), "oracle": best.tolist(),
            "hull_violation": safe}


def report(res):
    lines = ["# How much room is there? Oracle weights inside the shield", "",
             "Mean error on the Phase 3 benchmark instances (10 graphs per F, T = 2000). The oracle tunes a free "
             "score for every node and neighbour slot, per instance, knowing x*; it is the best any shielded "
             "defender could do against that attack. c is the floor constant of the shield.", "",
             "| F | Attack | LF | Oracle, c = 0.25 | Oracle, c = 0.05 | Room (c = 0.25) | Room (c = 0.05) |",
             "|---|---|---|---|---|---|---|"]
    for F in (1, 2):
        for a in ATTACKS:
            r = {x["c"]: x for x in res if x["F"] == F and x["attack"] == a}
            lf = np.mean(r[CS[0]]["lf"])
            o = {c: np.mean(r[c]["oracle"]) for c in CS}
            lines.append(f"| {F} | {a} | {lf:.3f} | {o[0.25]:.3f} | {o[0.05]:.3f} | "
                         f"{100 * (1 - o[0.25] / lf):.0f}% | {100 * (1 - o[0.05] / lf):.0f}% |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(F, a, c, 150) for F in (1, 2) for a in ATTACKS for c in CS]
    with get_context("spawn").Pool(6) as pool:
        res = pool.map(job, jobs, chunksize=1)
    (OUT / "room.json").write_text(json.dumps(res))
    md = report(res)
    (OUT / "room.md").write_text(md)
    print(md)
