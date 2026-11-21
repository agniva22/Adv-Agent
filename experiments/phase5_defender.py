"""Phase 5: the shielded RL defender and its ablations.

Three defenders are trained by backpropagation through the simulator on random
training graphs with F = 1 and F = 2 mixed:

  * Shielded RL: LF filter + floored softmax, trained
    against a mix of every scripted attack, no attack, and the Phase 4 RL
    attackers that were trained against LF;
  * Shielded RL (no adv. training): same shield, trained with no adversaries;
  * Unshielded RL: no filter and no floor (ablation), same training mix.

All are evaluated on the Phase 3 benchmark instances against every scripted
attack and the Phase 4 attacker (a transfer attack; Phases 6-7 train attackers
against the defenders themselves). Writes results/phase5/{table.md,
results.json, heatmap.png, training.png, defenders/*.pt}.
    PYTHONPATH=. python3 experiments/phase5_defender.py [--iters N]
"""

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from advagent import bench
from advagent import learn as Lr
from advagent import torchsim as ts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase5"
ATT_DIR = ROOT / "results" / "phase4" / "attackers"
BASE = ["LF", "LF-Metropolis", "Su-Vaidya"]
VARIANTS = {
    "Shielded RL": dict(shield=True, adversarial=True),
    "Shielded RL (no adv. training)": dict(shield=True, adversarial=False),
    "Unshielded RL": dict(shield=False, adversarial=True),
}


def load_attacker(F, defence="LF"):
    path = ATT_DIR / f"{defence.replace(' ', '_').replace('(', '').replace(')', '')}_F{F}_bias_random.pt"
    if not path.exists():
        return None
    pol = Lr.AttackerPolicy()
    pol.load_state_dict(torch.load(path, map_location="cpu"))
    return pol.eval()


def training_mix(adversarial, attackers):
    if not adversarial:
        return [(1.0, lambda F: None)]
    mix = [
        (1.0, lambda F: None),
        (0.5, lambda F: ts.Fixed(50.0)), (0.5, lambda F: ts.Fixed(-50.0)),
        (1.0, lambda F: ts.UniformRandom(-50, 50)),
        (0.5, lambda F: ts.FarthestEvader(+1, 0.9)), (0.5, lambda F: ts.FarthestEvader(-1, 0.9)),
        (1.0, lambda F: ts.Band(0.99)), (1.0, lambda F: ts.Band(-0.99)),
        (1.0, lambda F: ts.StallSwitcher()),
        (1.0, lambda F: "stealthy"),
    ]
    for F, pol in attackers.items():
        if pol is not None:
            for direc in (+1, -1):
                mix.append((1.0, lambda F_, p=pol, dr=direc, Fa=F: Lr.LearnedAttack(p, direc=dr) if F_ == Fa else ts.Band(0.99 * dr)))
    return mix


def train_variant(job):
    name, iters, T_train, device = job
    torch.set_num_threads(4)
    cfg = VARIANTS[name]
    attackers = {F: load_attacker(F) for F in (1, 2)}
    pools = {F: Lr.train_pool(F, 512) for F in (1, 2)}
    lines = []
    pol = Lr.ShieldedDefender(shield=cfg["shield"], name=name)
    pol, hist = Lr.train_defender(pol, pools, training_mix(cfg["adversarial"], attackers), T=T_train, iters=iters,
                                  device=device, log=lines.append)
    (OUT / "defenders").mkdir(parents=True, exist_ok=True)
    torch.save(pol.state_dict(), OUT / "defenders" / f"{name.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')}.pt")
    return {"name": name, "loss": hist, "log": lines}


def load_defender(name):
    cfg = VARIANTS[name]
    pol = Lr.ShieldedDefender(shield=cfg["shield"], name=name)
    pol.load_state_dict(torch.load(OUT / "defenders" / f"{name.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')}.pt", map_location="cpu"))
    return pol.eval()


def rl_attack_spec(F, attackers):
    """The Phase 4 attacker trained against LF, in the direction that hurts more (chosen per instance below)."""
    return attackers.get(F)


def eval_all(methods):
    """methods: {name: rule_factory}. Returns {(name, F, attack): metrics}."""
    attackers = {F: load_attacker(F) for F in (1, 2)}
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


def attacks_for(res, F):
    return [a for a in list(bench.SCRIPTED) + ["RL attacker (vs LF)"] if any(k[1] == F and k[2] == a for k in res)]


def table(res, methods, uw):
    lines = ["# Phase 5: shielded RL defender", "",
             "Mean error |x̄(T) − x*| on the Phase 3 benchmark instances (10 graphs per F, T = 2000). "
             "⚠ = some run unsafe (hull violation or α = 0 range excess above 1e−3); ∅ = some run without consensus (spread > 1). "
             "**Mean over attacks** excludes 'none'.", ""]
    for F in (1, 2):
        attacks = attacks_for(res, F)
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


def heatmap(res, methods, path, title):
    names = list(methods)
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(names) + 2.6), sharey=True)
    norm = matplotlib.colors.LogNorm(1e-2, 2)
    for ax, F in zip(axes, (1, 2)):
        attacks = attacks_for(res, F)
        E = np.array([[res[(n, F, a)]["error"].mean() for a in attacks] for n in names])
        im = ax.imshow(np.clip(E, 1e-2, None), cmap="Blues", norm=norm, aspect="auto")
        for i, n in enumerate(names):
            for j, a in enumerate(attacks):
                f = bench.flag(res[(n, F, a)])
                ax.text(j, i, f"{E[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="#1f1f1f" if norm(max(E[i, j], 1e-2)) < 0.6 else "white")
                if f:
                    ax.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False,
                                               ec="#d6311f" if f == "unsafe" else "#1f1f1f", lw=1.6,
                                               ls="-" if f == "unsafe" else "--"))
        ax.set_xticks(range(len(attacks)), attacks, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_title(f"F = {F}", fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("mean error |x̄ − x*| (log)")
    handles = [plt.Rectangle((0, 0), 1, 1, fill=False, ec="#d6311f", lw=1.6, label="unsafe in some run"),
               plt.Rectangle((0, 0), 1, 1, fill=False, ec="#1f1f1f", lw=1.6, ls="--", label="no consensus in some run")]
    fig.legend(handles=handles, loc="upper left", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.02, -0.1))
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def training_plot(train):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for r, c in zip(train, ["#2a78d6", "#1baf7a", "#eb6834"]):
        y = np.convolve(r["loss"], np.ones(20) / 20, mode="valid")
        ax.plot(y, color=c, lw=1.4, label=r["name"])
    ax.set(xlabel="training iteration", ylabel="loss relative to plain LF (1 = LF)\n(20-iteration moving average)",
           title="Phase 5: defender training")
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
    methods.update({n: (lambda n=n: load_defender(n)) for n in VARIANTS})
    res = eval_all(methods)
    uw = uw_eval(methods)
    (OUT / "results.json").write_text(json.dumps({"grid": summary(res), "uw": uw}))
    md = table(res, methods, uw)
    (OUT / "table.md").write_text(md)
    heatmap(res, methods, OUT / "heatmap.png", "Phase 5: error of the learned defenders vs. the safe baselines")
    print(md)
