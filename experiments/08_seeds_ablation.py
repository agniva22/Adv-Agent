"""Seeds, ablation and significance tests for memory + fallback (Table III, ablation table).

  memory + fallback (seed 1), (seed 2)   the defender of 04_memory_fallback.py with other
                                         training seeds (c = 0.05, diff loss, 300 iterations);
  memory only                            the trust memory without the LF fallback gate;
  fallback only                          the LF fallback gate without the memory.

Each gets its own best-response RL attacker (warm start from the LF attacker,
200 PPO iterations, seed 7, as in 06_evaluate.py) and is evaluated like 06_evaluate.py
on the 100 unseen graphs per F. LF, Trimmed mean, Su-Vaidya and memory + fallback
(seed 0) are read from results/evaluation/results.json (same graphs, same attacks).
The table adds paired Wilcoxon tests against LF and Trimmed mean, also for the
per-graph error averaged over the three seeds.
Writes results/seeds_ablation/{table.md, results.json, defenders/*.pt, attackers/*.pt}.
Finished steps are skipped, so an interrupted run can be restarted.

    python experiments/08_seeds_ablation.py [--iters N] [--br-iters N] [--jobs J] [--eval-only]
"""

import argparse
import json
from multiprocessing import get_context

import numpy as np
import torch
from scipy.stats import wilcoxon

from advagent import bench
from advagent import checkpoints as C
from advagent import learn as Lr

OUT = C.SEEDS_ABLATION
EVALUATION = C.EVALUATION / "results.json"
DEF = "memory + fallback"
BR = "RL attacker (best response)"
VARIANTS = {
    "memory + fallback (seed 1)": dict(memory=True, fallback=True, seed=1),
    "memory + fallback (seed 2)": dict(memory=True, fallback=True, seed=2),
    "memory only": dict(memory=True, fallback=False, seed=0),
    "fallback only": dict(memory=False, fallback=True, seed=0),
}
FROM_EVALUATION = ["LF", "Su-Vaidya", "Trimmed mean", DEF]
SEEDS = [DEF, "memory + fallback (seed 1)", "memory + fallback (seed 2)"]
UNSEEN = 100


def make(name):
    v = VARIANTS[name]
    if v["memory"]:
        pol = Lr.TrustDefender(c=0.05, name=name)
        if not v["fallback"]:
            pol.gate = None
    else:
        pol = Lr.ShieldedDefender(shield=True, c=0.05, name=name)
        h = pol.phi[0].out_features
        pol.gate = Lr.mlp(h, h, 1, depth=1)   # the fallback gate of TrustDefender
    return pol


def load(name):
    pol = make(name)
    pol.load_state_dict(torch.load(OUT / "defenders" / f"{C.slug(name)}.pt", map_location="cpu"))
    return pol.eval()


def train_job(job):
    name, iters, device = job
    path = OUT / "defenders" / f"{C.slug(name)}.pt"
    if path.exists():
        return
    torch.set_num_threads(2 if device == "cuda" else 4)
    pools = {F: Lr.train_pool(F, 512) for F in (1, 2)}
    mix = Lr.training_mix(True, {F: C.rl_attacker("LF", F) for F in (1, 2)})
    log = lambda s: open(OUT / f"train_{C.slug(name)}.log", "a").write(s + "\n")
    pol, _ = Lr.train_defender(make(name), pools, mix, T=2000, iters=iters, device=device, seed=VARIANTS[name]["seed"],
                               loss_type="diff", log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in pol.state_dict().items()}, path)


def br_job(job):
    name, F, iters, device = job
    path = OUT / "attackers" / f"{C.slug(name)}_F{F}.pt"
    if path.exists():
        return
    torch.set_num_threads(2 if device == "cuda" else 4)
    r = load(name).to(device)
    pol = C.rl_attacker("LF", F).train()   # warm start, as in 06_evaluate.py
    log = lambda s: open(OUT / f"train_br_{C.slug(name)}_F{F}.log", "a").write(s + "\n")
    pol, _ = Lr.train_attacker(lambda: r, Lr.train_pool(F, 512), iters=iters, policy=pol, seed=7, device=device, log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in pol.state_dict().items()}, path)


def eval_job(job):
    name, F = job
    torch.set_num_threads(3)
    insts = Lr.train_pool(F, UNSEEN, offset=300_000)   # the unseen graphs of 06_evaluate.py
    r = load(name)
    res = {}
    with torch.no_grad():
        for att, af in bench.SCRIPTED.items():
            res[att] = bench.evaluate(lambda: r, insts, lambda af=af: af(F))
        pol = Lr.AttackerPolicy()
        pol.load_state_dict(torch.load(OUT / "attackers" / f"{C.slug(name)}_F{F}.pt", map_location="cpu"))
        runs = [bench.evaluate(lambda: r, insts, lambda dr=dr: Lr.LearnedAttack(pol.eval(), direc=dr)) for dr in (1, -1)]
    pick = np.argmax([runs[0]["error"], runs[1]["error"]], 0)
    res[BR] = {k: np.where(pick == 0, runs[0][k], runs[1][k]) for k in runs[0]}
    print(f"  evaluated {name} F={F}", flush=True)
    return name, F, {a: {k: v.tolist() for k, v in m.items()} for a, m in res.items()}


def per_graph(res, name, F):
    """Per-graph errors: no attack, mean over attacks, own best response; plus safety."""
    m = res[(name, F)]
    atts = [a for a in m if a != "none"]
    e = {a: np.array(m[a]["error"]) for a in m}
    flags = {bench.flag({k: np.array(v) for k, v in m[a].items()}) for a in m} - {""}
    return {"none": e["none"], "mean": np.mean([e[a] for a in atts], 0), "br": e[BR], "safe": not flags}


def table(res):
    names = FROM_EVALUATION + list(VARIANTS)
    lines = ["# Seeds and ablations of memory + fallback", "",
             f"{UNSEEN} unseen graphs per F (those of 06_evaluate.py), T = 2000. Mean is over the scripted attacks and the "
             f"method's own best-response RL attacker; 'own BR' is that attacker alone. p: paired Wilcoxon test over the "
             "graphs (two-sided); 'better' counts graphs where the method has the lower error.", ""]
    for F in (1, 2):
        g = {n: per_graph(res, n, F) for n in names}
        lines += [f"## F = {F}", "", "| Method | No attack | Mean over attacks | Own BR | Safe? | Own BR vs LF: better, p | "
                  "Own BR vs Trimmed mean: better, p | Mean vs LF: better, p |", "|---|---|---|---|---|---|---|---|"]

        def cmp(a, b):
            return f"{int(np.sum(a < b))}/{a.size}, {wilcoxon(a, b).pvalue:.1e}"

        rows = list(names) + ["memory + fallback, 3 seeds averaged"]
        g[rows[-1]] = {k: np.mean([g[s][k] for s in SEEDS], 0) for k in ("none", "mean", "br")}
        g[rows[-1]]["safe"] = all(g[s]["safe"] for s in SEEDS)
        for n in rows:
            x = g[n]
            vs = "—" if n == "LF" else cmp(x["br"], g["LF"]["br"])
            vt = "—" if n == "Trimmed mean" else cmp(x["br"], g["Trimmed mean"]["br"])
            vm = "—" if n == "LF" else cmp(x["mean"], g["LF"]["mean"])
            lines.append(f"| {n} | {x['none'].mean():.3f} | {x['mean'].mean():.3f} | {x['br'].mean():.3f} | "
                         f"{'yes' if x['safe'] else 'no'} | {vs} | {vt} | {vm} |")
        sd = {k: np.std([g[s][k].mean() for s in SEEDS]) for k in ("none", "mean", "br")}
        mu = {k: np.mean([g[s][k].mean() for s in SEEDS]) for k in ("none", "mean", "br")}
        lines += ["", f"Over the 3 seeds of memory + fallback (mean ± std of the seed means): no attack "
                  f"{mu['none']:.3f} ± {sd['none']:.3f}, mean over attacks {mu['mean']:.3f} ± {sd['mean']:.3f}, "
                  f"own BR {mu['br']:.3f} ± {sd['br']:.3f}.", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--br-iters", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=2, help="trainings run at the same time")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        with get_context("spawn").Pool(args.jobs) as pool:
            pool.map(train_job, [(n, args.iters, args.device) for n in VARIANTS], chunksize=1)
            pool.map(br_job, [(n, F, args.br_iters, args.device) for n in VARIANTS for F in (1, 2)], chunksize=1)
    with get_context("spawn").Pool(3) as pool:   # memory defenders keep per-step memory: RAM
        out = pool.map(eval_job, [(n, F) for n in VARIANTS for F in (1, 2)], chunksize=1)
    res = {(n, F): r for n, F, r in out}
    grid = json.loads(EVALUATION.read_text())["grid"]
    for n in FROM_EVALUATION:
        for F in (1, 2):
            res[(n, F)] = {key.split("|")[2]: v for key, v in grid.items() if key.startswith(f"{n}|F{F}|")}
    (OUT / "results.json").write_text(json.dumps({f"{n}|F{F}": r for (n, F), r in res.items() if n in VARIANTS}))
    md = table(res)
    (OUT / "table.md").write_text(md)
    print(md)
