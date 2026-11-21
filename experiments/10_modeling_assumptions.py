"""How much do the results depend on the modelling assumptions of the paper?

Every policy (defenders and best-response attackers) is used exactly as trained by
02-06, on 20-node graphs of one random family, with synchronous updates, fixed
links, known F, f_i = (x - a_i)^2 with a_i in [-5, 5], and alpha_t = 0.5/(t+1)^0.6.
Here each assumption is changed in turn, nothing is retrained:

  reference              the setting of 06_evaluate.py (fresh graphs), for comparison
  sparse graphs          each new node gets exactly 2F+1 neighbours (extra = 0)
  dense graphs           training family plus random extra edges (p = 0.2)
  hub graphs             preferential attachment: new nodes prefer high-degree nodes
  heterogeneous costs    f_i = s_i (x - a_i)^2 with s_i in [0.5, 2]
  wider range            a_i and x_i(0) in [-20, 20]
  link failures 10% / 30%   every link fails independently at every step
  asynchronous 80% / 50%    every regular agent updates at a step with this probability
  F overestimated        adversaries are F-local but agents filter with F + 1
  F underestimated       graph and adversaries for F = 2, agents filter with F = 1 (F = 1 row only)
  other step sizes       alpha_t = 1/(t+1)^0.75
  longer horizon         T = 5000

The graph families are (2F+1)-robust by construction (each added node has at least
2F+1 neighbours; adding edges keeps robustness). Link failures and asynchronous
updates are outside the theory of [1]; the F settings break its assumptions on
purpose. Safety is checked as in 06_evaluate.py (hull violation for t >= T/2 and the
alpha = 0 range check, both above 1e-3); 'no consensus' if the final spread > 1.

Writes results/modeling_assumptions/{table.md, results.json, parts/*.json}. Finished
parts are skipped, so an interrupted run can be restarted.

    python experiments/10_modeling_assumptions.py [--settings ...] [--graphs N] [--workers W]
"""

import argparse
import json
import os
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from advagent import bench
from advagent import checkpoints as C
from advagent import graphs as g
from advagent import learn as Lr
from advagent import torchsim as ts

OUT = Path(os.environ.get("ADV_MODELING_OUT", C.MODELING_ASSUMPTIONS))   # override for smoke tests
BR = "RL attacker (best response)"
NAMES = ["LF", "LF-Metropolis", "Su-Vaidya", "Trimmed mean", "memory + fallback", "memory + fallback (game)"]
ATTACKS = ["none", "fixed +50", "random Byzantine", "farthest evader", "stealthy (fake 5)", "stealthy (fake 50)",
           "trim-hugger", "switching"]
CHUNK = 10   # graphs per simulator batch (the memory defenders keep per-step memory: RAM)
TOL, SPREAD_TOL = 1e-3, 1.0


# ------------------------------------------------------------------ instances

def hub_graph(n, r, rng, extra=1):
    """Like ``graphs.random_robust``, but each new node picks its r + extra
    neighbours with probability proportional to their degree (hubs form).
    Every added node has >= r neighbours, so the graph stays r-robust."""
    m0 = max(2 * r - 1, r + extra + 1)
    adj = np.zeros((n, n), dtype=bool)
    adj[:m0, :m0] = True
    np.fill_diagonal(adj, False)
    for v in range(m0, n):
        deg = adj[:v, :v].sum(1).astype(float)
        nbrs = rng.choice(v, size=r + extra, replace=False, p=deg / deg.sum())
        adj[v, nbrs] = adj[nbrs, v] = True
    return adj


def make_instance(setting, seed, F):
    rng = np.random.default_rng(seed)
    n, lo, hi = 20, -5.0, 5.0
    F_graph, F_filter = F, F
    if setting == "F underestimated":
        F_graph, F_filter = 2, 1
    if setting == "F overestimated":
        F_filter = F + 1
    r = 2 * F_graph + 1
    if setting == "sparse graphs":
        adj = g.random_robust(n, r, rng, extra=0)
    elif setting == "dense graphs":
        adj = g.random_robust(n, r, rng, extra=1) | g.erdos_renyi(n, 0.2, rng)
        np.fill_diagonal(adj, False)
    elif setting == "hub graphs":
        adj = hub_graph(n, r, rng, extra=1)
    else:
        adj = g.random_robust(n, r, rng, extra=1)
    adv = g.random_f_local_set(adj, F_graph, 3 * F_graph, rng)
    if setting == "wider range":
        lo, hi = -20.0, 20.0
    I = {"adj": adj, "adv": adv, "a": rng.uniform(lo, hi, n), "x0": rng.uniform(lo, hi, n), "F": F_filter}
    if setting == "heterogeneous costs":
        I["s"] = rng.uniform(0.5, 2.0, n)
    return I


SETTINGS = {   # name -> (T, alpha, link failure probability, update probability)
    "reference": (2000, None, 0.0, 1.0),
    "sparse graphs": (2000, None, 0.0, 1.0),
    "dense graphs": (2000, None, 0.0, 1.0),
    "hub graphs": (2000, None, 0.0, 1.0),
    "heterogeneous costs": (2000, None, 0.0, 1.0),
    "wider range": (2000, None, 0.0, 1.0),
    "link failures 10%": (2000, None, 0.1, 1.0),
    "link failures 30%": (2000, None, 0.3, 1.0),
    "asynchronous 80%": (2000, None, 0.0, 0.8),
    "asynchronous 50%": (2000, None, 0.0, 0.5),
    "F overestimated": (2000, None, 0.0, 1.0),
    "F underestimated": (2000, None, 0.0, 1.0),
    "other step sizes": (2000, lambda t: 1.0 / (t + 1.0) ** 0.75, 0.0, 1.0),
    "longer horizon": (5000, None, 0.0, 1.0),
}


def f_values(setting):
    return (1,) if setting == "F underestimated" else (1, 2)


# ----------------------------------------------------------------- simulation

@torch.no_grad()
def simulate(batch, rule, T, alpha, attack, drop, q, seed):
    """``ts.run`` with optional link failures and asynchronous updates.
    Returns per-instance error, spread, hull violation (t >= T/2) and the range
    of the regular states over the whole run (for the alpha = 0 check)."""
    gen = torch.Generator().manual_seed(seed)
    base = (batch.nmask.clone(), batch.from_adv.clone(), batch.n_adv_in.clone(), batch.deg.clone(), batch.nbr_deg.clone())
    rule.reset(batch)
    astate = attack.init(batch) if attack is not None else ()
    x = batch.x0.clone()
    big = torch.finfo(x.dtype).max
    lo_all = torch.where(batch.ev, x, big).amin(1)
    hi_all = torch.where(batch.ev, x, -big).amax(1)
    lo = torch.full_like(lo_all, big)
    hi = -lo
    try:
        for t in range(T):
            if drop > 0:   # this step's links; the rules and attacks see only these
                up = torch.rand(base[0].shape, generator=gen) >= drop
                batch.nmask = base[0] & up
                batch.from_adv = base[1] & up
                batch.n_adv_in = batch.from_adv.sum(-1)
                batch.deg = batch.nmask.sum(-1).to(x.dtype)
                batch.nbr_deg = batch.gather(batch.deg)
            vals = batch.gather(x)
            if attack is not None:
                av, astate = attack(t, x, vals, batch, astate)
                vals = torch.where(batch.from_adv, av, vals)
            new = ts.descend(rule, x, vals, batch, t, alpha(t), attack is not None, getattr(attack, "grads", None))
            upd = batch.reg if q >= 1 else batch.reg & (torch.rand(x.shape, generator=gen) < q)
            x = torch.where(upd, new, x)
            xr_lo = torch.where(batch.ev, x, big).amin(1)
            xr_hi = torch.where(batch.ev, x, -big).amax(1)
            lo_all, hi_all = torch.minimum(lo_all, xr_lo), torch.maximum(hi_all, xr_hi)
            if t + 1 >= T // 2:
                lo, hi = torch.minimum(lo, xr_lo), torch.maximum(hi, xr_hi)
    finally:
        batch.nmask, batch.from_adv, batch.n_adv_in, batch.deg, batch.nbr_deg = base
    xr = torch.where(batch.ev, x, torch.nan)
    return {
        "error": (batch.mean(x) - batch.x_star).abs(),
        "spread": xr.nan_to_num(-big).amax(1) - xr.nan_to_num(big).amin(1),
        "hull_violation": torch.clamp(torch.maximum(batch.hull_lo - lo, hi - batch.hull_hi), min=0),
        "lo_all": lo_all, "hi_all": hi_all,
    }


def evaluate(make_rule, insts, make_attack, T, alpha, drop, q):
    """Per-instance metrics, in chunks of CHUNK graphs, plus the alpha = 0 range check."""
    out = {"error": [], "spread": [], "hull_violation": [], "excess": []}
    for i in range(0, len(insts), CHUNK):
        part = insts[i:i + CHUNK]
        run_insts, att = bench.prepare(part, make_attack())
        batch = ts.Batch(run_insts)
        torch.manual_seed(i)
        r = simulate(batch, make_rule(), T, alpha, att, drop, q, seed=1000 + i)
        _, att0 = bench.prepare(part, make_attack())
        z = simulate(batch, make_rule(), 300, lambda t: 0.0, att0, drop, q, seed=2000 + i)
        lo0 = torch.where(batch.ev, batch.x0, torch.finfo(batch.x0.dtype).max).amin(1)
        hi0 = torch.where(batch.ev, batch.x0, -torch.finfo(batch.x0.dtype).max).amax(1)
        excess = torch.clamp(torch.maximum(lo0 - z["lo_all"], z["hi_all"] - hi0), min=0)
        for k in ("error", "spread", "hull_violation"):
            out[k].append(r[k].numpy())
        out["excess"].append(excess.numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def part_path(setting, F, name):
    key = "".join(ch if ch.isalnum() else "_" for ch in f"{setting}_F{F}_{name}")
    return OUT / "parts" / f"{key}.json"


def eval_job(job):
    setting, F, name, graphs = job
    path = part_path(setting, F, name)
    if path.exists():
        return
    torch.set_num_threads(4)
    T, alpha, drop, q = SETTINGS[setting]
    alpha = alpha or ts.power_alpha()
    idx = list(SETTINGS).index(setting)
    insts = [make_instance(setting, 600_000 + 10_000 * idx + 1000 * F + s, F) for s in range(graphs)]
    rule = ts.BASELINES[name]() if name in ts.BASELINES else C.defender(name)
    res = {}
    for att in ATTACKS:
        af = bench.SCRIPTED[att]
        res[att] = evaluate(lambda: rule, insts, lambda af=af: af(insts[0]["F"]), T, alpha, drop, q)
    pol = C.best_response_attacker(name, min(insts[0]["F"], 2))
    runs = [evaluate(lambda: rule, insts, lambda dr=dr: Lr.LearnedAttack(pol, T=T, direc=dr), T, alpha, drop, q)
            for dr in (1, -1)]
    pick = np.argmax([runs[0]["error"], runs[1]["error"]], 0)
    res[BR] = {k: np.where(pick == 0, runs[0][k], runs[1][k]) for k in runs[0]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({a: {k: v.tolist() for k, v in m.items()} for a, m in res.items()}))
    print(f"  done: {setting}, F={F}, {name}", flush=True)


# --------------------------------------------------------------------- report

def flag(m):
    if (np.array(m["hull_violation"]) > TOL).any() or (np.array(m["excess"]) > TOL).any():
        return "unsafe"
    if (np.array(m["spread"]) > SPREAD_TOL).any():
        return "no consensus"
    return ""


def summary(res):
    """mean over attacks (excluding 'none'), best-response error, flags."""
    att = [a for a in res if a != "none"]
    flags = sorted({flag(res[a]) for a in res} - {""})
    return float(np.mean([np.mean(res[a]["error"]) for a in att])), float(np.mean(res[BR]["error"])), flags


def table(settings, graphs):
    lines = ["# Changing the modelling assumptions", "",
             f"Mean error over {graphs} random graphs per setting and F, nothing retrained. Each cell: mean over the "
             f"attacks (except 'none') / error under the method's own best-response RL attacker. ⚠ unsafe, ∅ no consensus "
             "(under some attack). Best safe value per row in bold.", ""]
    data = {}
    for F in (1, 2):
        rows = [s for s in settings if F in f_values(s)]
        lines += [f"## F = {F}", "", "| Setting | " + " | ".join(NAMES) + " |", "|---" * (len(NAMES) + 1) + "|"]
        for s in rows:
            cells = {}
            for name in NAMES:
                path = part_path(s, F, name)
                if path.exists():
                    res = json.loads(path.read_text())
                    cells[name] = summary(res)
                    data[f"{s}|F{F}|{name}"] = cells[name]
            safe = [n for n in cells if not cells[n][2]]
            best_m = min((cells[n][0] for n in safe), default=None)
            best_b = min((cells[n][1] for n in safe), default=None)
            out = []
            for name in NAMES:
                if name not in cells:
                    out.append("—")
                    continue
                m, b, fl = cells[name]
                ms = f"**{m:.3f}**" if m == best_m else f"{m:.3f}"
                bs = f"**{b:.3f}**" if b == best_b else f"{b:.3f}"
                mark = (" ⚠" if "unsafe" in fl else "") + (" ∅" if "no consensus" in fl else "")
                out.append(f"{ms} / {bs}{mark}")
            lines.append(f"| {s} | " + " | ".join(out) + " |")
        lines.append("")
    return "\n".join(lines) + "\n", data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS), metavar="SETTING")
    ap.add_argument("--graphs", type=int, default=50)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, F, name, args.graphs) for s in args.settings for F in f_values(s) for name in NAMES]
    with get_context("spawn").Pool(args.workers) as pool:
        pool.map(eval_job, jobs, chunksize=1)
    md, data = table(args.settings, args.graphs)
    (OUT / "table.md").write_text(md)
    (OUT / "results.json").write_text(json.dumps(data))
    print(md)
