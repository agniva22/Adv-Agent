"""Closing the switching gap: memory + fallback variants trained with more switching attacks and/or a longer memory.

Su–Vaidya is still slightly better than the memory + fallback defender against the
switching attack. Three variants, each trained like memory + fallback (c = 0.05,
diff loss, 300 iterations):

  switching-heavy mix     same defender; extra switching attacks in training (stall-switchers
                          with windows 25 / 50 / 100 and flickers with periods 25 / 50);
  two-timescale memory    memory with factors 0.95 (~20 steps) and 0.995 (~200 steps);
  both                    the two together.

All are compared on 100 unseen graphs per F with the attacks of ``defender_br.py``.
Writes results/defender_switching/{switching.md, switching.json, training.json, defenders/*.pt}.
    PYTHONPATH=. python experiments/defender_switching.py [--iters N] [--eval-only]
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
OUT = ROOT / "results" / "defender_switching"
DEF = "memory + fallback"
VARIANTS = {
    "switching-heavy mix": dict(betas=(0.95,), heavy=True),
    "two-timescale memory": dict(betas=(0.95, 0.995), heavy=False),
    "two-timescale + switching-heavy": dict(betas=(0.95, 0.995), heavy=True),
}
UNSEEN = 100


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fname(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


def make(name):
    return Lr.TrustDefender(c=0.05, betas=VARIANTS[name]["betas"], name=name)


def mix(heavy):
    p5 = module("phase5_defender")
    m = p5.training_mix(True, {F: p5.load_attacker(F) for F in (1, 2)})
    if heavy:
        Sleeper = module("defender_br").Sleeper
        m += [(1.0, lambda F, w=w: ts.StallSwitcher(window=w)) for w in (25, 50, 100)]
        m += [(0.5, lambda F, p=p, a=a: Sleeper(period=p, a=a)) for p in (25, 50) for a in (0.99, -0.99)]
    return m


def train_job(job):
    name, iters, device = job
    path = OUT / "defenders" / f"{fname(name)}.pt"
    if path.exists():   # trained by an earlier (crashed) run
        return {"name": name, "loss": []}
    torch.set_num_threads(2 if device == "cuda" else 4)
    pools = {F: Lr.train_pool(F, 512) for F in (1, 2)}
    log = lambda s: open(OUT / f"train_{fname(name)}.log", "a").write(s + "\n")
    pol, hist = Lr.train_defender(make(name), pools, mix(VARIANTS[name]["heavy"]), T=2000, iters=iters,
                                  device=device, loss_type="diff", log=log)
    (OUT / "defenders").mkdir(parents=True, exist_ok=True)
    torch.save(pol.state_dict(), OUT / "defenders" / f"{fname(name)}.pt")
    return {"name": name, "loss": hist}


def load(name):
    pol = make(name)
    pol.load_state_dict(torch.load(OUT / "defenders" / f"{fname(name)}.pt", map_location="cpu"))
    return pol.eval()


def rule(name):
    if name == DEF:
        return module("defender_trust").load(DEF)
    if name in VARIANTS:
        return load(name)
    return ts.BASELINES[name]()


def eval_job(job):
    name, F = job
    torch.set_num_threads(4)
    br = module("defender_br")
    insts = Lr.train_pool(F, UNSEEN, offset=300_000)
    r = rule(name)
    res = {}
    with torch.no_grad():
        for att, af in br.ATTACKS.items():
            res[att] = bench.evaluate(lambda: r, insts, lambda af=af: af(F))
        lf_att = br.attacker(br.P4 / f"LF_F{F}_bias_random.pt")
        runs = [bench.evaluate(lambda: r, insts, lambda dr=dr: Lr.LearnedAttack(lf_att, direc=dr)) for dr in (1, -1)]
    pick = np.argmax([runs[0]["error"], runs[1]["error"]], 0)
    res["RL attacker (vs LF)"] = {k: np.where(pick == 0, runs[0][k], runs[1][k]) for k in runs[0]}
    return name, F, {a: {k: v.tolist() for k, v in m.items()} for a, m in res.items()}


def table(res, names):
    lines = ["# Closing the switching gap", "",
             f"Mean error over {UNSEEN} unseen graphs per F (T = 2000). Mean is over all attacks except 'none'; "
             "'Wins vs Su–Vaidya' counts attacks (with 'none') on which the method beats Su–Vaidya.", ""]
    for F in (1, 2):
        atts = list(res[(names[0], F)])
        err = {n: {a: np.mean(res[(n, F)][a]["error"]) for a in atts} for n in names}
        lines += [f"## F = {F}", "", "| Method | No attack | Switching | Mean over attacks | Worst attack | Wins vs Su–Vaidya | Safe? |",
                  "|---|---|---|---|---|---|---|"]
        for n in names:
            att = {a: e for a, e in err[n].items() if a != "none"}
            worst = max(att, key=att.get)
            wins = sum(err[n][a] < err["Su-Vaidya"][a] for a in atts)
            flags = {bench.flag({k: np.array(v) for k, v in res[(n, F)][a].items()}) for a in atts} - {""}
            lines.append(f"| {n} | {err[n]['none']:.3f} | {err[n]['switching']:.3f} | {np.mean(list(att.values())):.3f} | "
                         f"{att[worst]:.3f} ({worst}) | {wins}/{len(atts)} | {'yes' if not flags else ', '.join(sorted(flags))} |")
        lines += ["", "| Attack | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        lines += [f"| {a} | " + " | ".join(f"{err[n][a]:.3f}" for n in names) + " |" for a in atts]
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        with get_context("spawn").Pool(len(VARIANTS)) as pool:
            train = pool.map(train_job, [(n, args.iters, args.device) for n in VARIANTS], chunksize=1)
        (OUT / "training.json").write_text(json.dumps(train))
    names = ["LF", "Su-Vaidya", DEF] + list(VARIANTS)
    with get_context("spawn").Pool(3) as pool:   # 6 ran out of RAM next to the Phase 6 game
        out = pool.map(eval_job, [(n, F) for n in names for F in (1, 2)], chunksize=1)
    res = {(n, F): r for n, F, r in out}
    (OUT / "switching.json").write_text(json.dumps({f"{n}|F{F}": r for (n, F), r in res.items()}))
    md = table(res, names)
    (OUT / "switching.md").write_text(md)
    print(md)
