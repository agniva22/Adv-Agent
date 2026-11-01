"""Fixes 1-4 for the shielded defender, compared on the Phase 5 benchmark.

Phase 5's defender stayed at LF. Each variant here changes one more thing:

  fix 1  'diff' loss: batch loss minus LF's, over LF's (no per-episode ratio);
  fix 4  smaller floor c = 0.05 (more freedom; the shield proof holds for any c > 0);
  fix 2  two stages: first no attacks (learn to undo LF's averaging bias), then the full mix;
  fix 3  truncated BPTT: a loss every 200 steps instead of only at T.

Writes results/defender_fixes/{fixes.md, fixes.json, training.png, defenders/*.pt}.
    PYTHONPATH=. python experiments/defender_fixes.py [--iters N]
"""

import argparse
import importlib.util
import json
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from advagent import learn as Lr
from advagent import torchsim as ts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "defender_fixes"
VARIANTS = {
    "fix 1: diff loss": dict(c=0.25, loss_type="diff"),
    "fix 1+4: diff loss, c = 0.05": dict(c=0.05, loss_type="diff"),
    "fix 1+4+2: two stages": dict(c=0.05, loss_type="diff", stages=True),
    "fix 1+4+3: truncated BPTT": dict(c=0.05, loss_type="diff", window=200),
}


def phase5():
    spec = importlib.util.spec_from_file_location("p5", ROOT / "experiments" / "phase5_defender.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fname(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


def make(name):
    return Lr.ShieldedDefender(shield=True, c=VARIANTS[name]["c"], name=name)


def train_job(job):
    name, iters, device = job
    torch.set_num_threads(2)
    cfg = VARIANTS[name]
    p5 = phase5()
    pools = {F: Lr.train_pool(F, 512) for F in (1, 2)}
    mix = p5.training_mix(True, {F: p5.load_attacker(F) for F in (1, 2)})
    kw = dict(T=2000, device=device, loss_type=cfg["loss_type"], window=cfg.get("window"))
    pol = make(name)
    if cfg.get("stages"):
        pol, h1 = Lr.train_defender(pol, pools, p5.training_mix(False, {}), iters=iters // 2, seed=0, **kw)
        pol, h2 = Lr.train_defender(pol, pools, mix, iters=iters - iters // 2, seed=1, lr=1e-3, **kw)
        hist = h1 + h2
    else:
        pol, hist = Lr.train_defender(pol, pools, mix, iters=iters, **kw)
    (OUT / "defenders").mkdir(parents=True, exist_ok=True)
    torch.save(pol.state_dict(), OUT / "defenders" / f"{fname(name)}.pt")
    return {"name": name, "loss": hist}


def load(name):
    pol = make(name)
    pol.load_state_dict(torch.load(OUT / "defenders" / f"{fname(name)}.pt", map_location="cpu"))
    return pol.eval()


def summary(res, names):
    p5 = phase5()
    lines = ["# Defender fixes", "",
             "Mean error on the Phase 3 benchmark instances (10 graphs per F, T = 2000), same attacks as Phase 5 "
             "(eight scripted attacks and the Phase 4 RL attacker trained against LF). Mean is over all attacks "
             "except 'none'. ⚠ / ∅ = some run unsafe / without consensus.", ""]
    for F in (1, 2):
        atts = p5.attacks_for(res, F)
        lines += [f"## F = {F}", "", "| Method | No attack | Mean over attacks | Worst attack | Change vs LF (mean) | Safe? |",
                  "|---|---|---|---|---|---|"]
        lf = np.mean([res[("LF", F, a)]["error"].mean() for a in atts if a != "none"])
        for n in names:
            errs = {a: res[(n, F, a)]["error"].mean() for a in atts if a != "none"}
            mean = np.mean(list(errs.values()))
            worst = max(errs, key=errs.get)
            flags = {p5.bench.flag(res[(n, F, a)]) for a in atts} - {""}
            lines.append(f"| {n} | {res[(n, F, 'none')]['error'].mean():.3f} | {mean:.3f} | "
                         f"{errs[worst]:.3f} ({worst}) | {100 * (mean / lf - 1):+.1f}% | "
                         f"{'yes' if not flags else ', '.join(sorted(flags))} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def training_plot(train):
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    for r, c in zip(train, ["#2a78d6", "#1baf7a", "#eb6834", "#7a4fd6"]):
        y = np.convolve(r["loss"], np.ones(20) / 20, mode="valid")
        ax.plot(y, color=c, lw=1.4, label=r["name"])
    ax.axhline(0, color="#888", lw=0.8, ls=":")
    ax.set(xlabel="training iteration", ylabel="(loss − LF loss) / LF loss\n(20-iteration moving average)",
           title="Defender fixes: training (below 0 = better than LF)")
    ax.legend(fontsize=7, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    training_plot(json.loads((OUT / "training.json").read_text()))
    torch.set_num_threads(8)
    p5 = phase5()
    methods = {"LF": lambda: ts.LF(), "Su-Vaidya": lambda: ts.BASELINES["Su-Vaidya"](),
               "Shielded RL (Phase 5)": lambda: p5.load_defender("Shielded RL")}
    methods.update({n: (lambda n=n: load(n)) for n in VARIANTS})
    res = p5.eval_all(methods)
    (OUT / "fixes.json").write_text(json.dumps(p5.summary(res)))
    md = summary(res, list(methods))
    (OUT / "fixes.md").write_text(md)
    print(md)
