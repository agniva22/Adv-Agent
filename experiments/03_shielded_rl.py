"""The shielded RL defender without memory, and the unshielded ablation.

Three defenders are trained by backpropagation through the simulator on random
training graphs with F = 1 and F = 2 mixed:

  * Shielded RL: LF filter + floored softmax, trained against every scripted
    attack, no attack, and the RL attacker of 02_rl_attackers.py trained against LF;
  * Shielded RL (no adv. training): the same shield, trained without adversaries;
  * Unshielded RL: no filter and no floor, the same training mix.

They are evaluated on the benchmark graphs against every scripted attack and the
LF attacker (a transfer attack; 06_evaluate.py trains an attacker against each
defender itself). Writes results/shielded_rl/{table.md, results.json, heatmap.png,
training.png, defenders/*.pt}.

    python experiments/03_shielded_rl.py [--iters N]
"""

import argparse
import json
from multiprocessing import get_context

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from advagent import bench
from advagent import checkpoints as C
from advagent import plots
from advagent import learn as Lr
from advagent import torchsim as ts

OUT = C.SHIELDED_RL
VARIANTS = C.SHIELDED_VARIANTS
BASE = ["LF", "LF-Metropolis", "Su-Vaidya"]



def train_variant(job):
    name, iters, T_train, device = job
    torch.set_num_threads(4)
    cfg = VARIANTS[name]
    attackers = {F: C.rl_attacker("LF", F) for F in (1, 2)}
    pools = {F: Lr.train_pool(F, 512) for F in (1, 2)}
    lines = []
    pol = Lr.ShieldedDefender(shield=cfg["shield"], name=name)
    pol, hist = Lr.train_defender(pol, pools, Lr.training_mix(cfg["adversarial"], attackers), T=T_train, iters=iters,
                                  device=device, log=lines.append)
    (OUT / "defenders").mkdir(parents=True, exist_ok=True)
    torch.save(pol.state_dict(), OUT / "defenders" / f"{C.slug(name)}.pt")
    return {"name": name, "loss": hist, "log": lines}


def eval_all(methods):
    """methods: {name: rule_factory}. Returns {(name, F, attack): metrics}."""
    attackers = {F: C.rl_attacker("LF", F) for F in (1, 2)}
    res = {}
    for F in (1, 2):
        insts = Lr.eval_instances(F)
        for name, fac in methods.items():
            for att, af in bench.SCRIPTED.items():
                res[(name, F, att)] = bench.evaluate(fac, insts, lambda af=af, F=F: af(F))
            if attackers[F] is not None:
                runs = [bench.evaluate(fac, insts, lambda dr=dr: Lr.LearnedAttack(attackers[F], direc=dr)) for dr in (1, -1)]
                pick = np.argmax([runs[0]["error"], runs[1]["error"]], 0)
                res[(name, F, "RL attacker (vs LF)")] = {k: np.where(pick == 0, runs[0][k], runs[1][k]) for k in runs[0]}
    return res


def uw_eval(methods):
    out = {}
    for name, fac in methods.items():
        m = bench.evaluate(fac, [bench.uw_instance(5)], lambda: None, T=4000)
        out[name] = float(m["error"][0])
    return out


def table(res, methods, uw):
    lines = ["# Shielded RL defender", "",
             "Mean error |x̄(T) − x*| on the benchmark graphs (10 graphs per F, T = 2000). "
             "⚠ = some run unsafe (hull violation or α = 0 range excess above 1e−3); ∅ = some run without consensus (spread > 1). "
             "**Mean over attacks** excludes 'none'.", ""]
    for F in (1, 2):
        attacks = plots.attacks_for(res, F)
        lines += [f"## F = {F}", "", "| Method | " + " | ".join(attacks) + " | Mean over attacks |",
                  "|---" * (len(attacks) + 2) + "|"]
        for name in methods:
            cells, means = [], []
            for a in attacks:
                m = res[(name, F, a)]
                cells.append(f"{m['error'].mean():.3f}{bench.MARK[bench.flag(m)]}")
                if a != "none":
                    means.append(m["error"].mean())
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {np.mean(means):.3f} |")
        lines.append("")
    lines += ["## U–W graph (Ex. 8.2, K = 5, b = 4, no adversaries; bound b/4 = 1)", "",
              "| Method | Error |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in uw.items()]
    return "\n".join(lines) + "\n"


def training_plot(train):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for r, c in zip(train, ["#2a78d6", "#1baf7a", "#eb6834"]):
        y = np.convolve(r["loss"], np.ones(20) / 20, mode="valid")
        ax.plot(y, color=c, lw=1.4, label=r["name"])
    ax.set(xlabel="training iteration", ylabel="loss relative to plain LF (1 = LF)\n(20-iteration moving average)",
           title="Shielded RL: defender training")
    ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def summary(res):
    return {f"{k[0]}|F{k[1]}|{k[2]}": {m: v.tolist() for m, v in d.items()} for k, d in res.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--T-train", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        with get_context("spawn").Pool(len(VARIANTS)) as pool:
            train = pool.map(train_variant, [(n, args.iters, args.T_train, args.device) for n in VARIANTS])
        (OUT / "training.json").write_text(json.dumps(train))
    train = json.loads((OUT / "training.json").read_text())
    training_plot(train)
    torch.set_num_threads(8)
    methods = {n: (lambda n=n: ts.BASELINES[n]()) for n in BASE}
    methods.update({n: (lambda n=n: C.defender(n)) for n in VARIANTS})
    res = eval_all(methods)
    uw = uw_eval(methods)
    (OUT / "results.json").write_text(json.dumps({"grid": summary(res), "uw": uw}))
    md = table(res, methods, uw)
    (OUT / "table.md").write_text(md)
    plots.heatmap(res, methods, OUT / "heatmap.png", "Error of the learned defenders vs. the safe baselines")
    print(md)
