"""Train the memory + fallback defender (Section 3 of the paper).

The defender keeps a per-neighbour trust memory and an LF fallback gate inside the
shield. It is trained by backpropagation through the simulator against the scripted
attacks and the RL attacker against LF, then compared with LF and Su-Vaidya on the
benchmark graphs and on 100 unseen graphs per F.
Writes results/memory_fallback/{trust.md, trust.json, training.json, training.png,
defenders/memory_fallback.pt}.

    python experiments/04_memory_fallback.py [--iters N] [--eval-only]
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
from advagent import learn as Lr
from advagent import torchsim as ts

OUT = C.MEMORY_FALLBACK
VARIANTS = {
    "memory + fallback": dict(trust=True, loss_type="diff"),
}
UNSEEN = 100


def make(name):
    if VARIANTS[name]["trust"]:
        return Lr.TrustDefender(c=0.05, name=name)
    return Lr.ShieldedDefender(shield=True, c=0.05, name=name)


def train_job(job):
    name, iters, device = job
    torch.set_num_threads(2)
    pools = {F: Lr.train_pool(F, 512) for F in (1, 2)}
    mix = Lr.training_mix(True, {F: C.rl_attacker("LF", F) for F in (1, 2)})
    pol, hist = Lr.train_defender(make(name), pools, mix, T=2000, iters=iters, device=device,
                                  loss_type=VARIANTS[name]["loss_type"])
    (OUT / "defenders").mkdir(parents=True, exist_ok=True)
    torch.save(pol.state_dict(), OUT / "defenders" / f"{C.slug(name)}.pt")
    return {"name": name, "loss": hist}


def load(name):
    pol = make(name)
    pol.load_state_dict(torch.load(OUT / "defenders" / f"{C.slug(name)}.pt", map_location="cpu"))
    return pol.eval()


def eval_on(methods, insts_for, device):
    """Like 03_shielded_rl.eval_all, on any instance set: {(name, F, attack): metrics}."""
    res = {}
    for F in (1, 2):
        insts, rl = insts_for(F), C.rl_attacker("LF", F)
        for name, fac in methods.items():
            with torch.no_grad():
                for att, af in bench.SCRIPTED.items():
                    res[(name, F, att)] = bench.evaluate(fac, insts, lambda af=af, F=F: af(F), device=device)
                runs = [bench.evaluate(fac, insts, lambda dr=dr: Lr.LearnedAttack(rl.to(device), direc=dr), device=device)
                        for dr in (1, -1)]
            pick = np.argmax([runs[0]["error"], runs[1]["error"]], 0)
            res[(name, F, "RL attacker (vs LF)")] = {k: np.where(pick == 0, runs[0][k], runs[1][k]) for k in runs[0]}
            print(f"  evaluated {name} F={F} on {len(insts)} graphs", flush=True)
    return res


def table(res, names, title, n_graphs):
    atts = list(bench.SCRIPTED) + ["RL attacker (vs LF)"]
    lines = [f"## {title}", "",
             f"Mean error over {n_graphs} graphs per F (T = 2000). Mean is over the 9 attacks (not 'none'). "
             "Wins = attacks on which the method beats LF.", ""]
    for F in (1, 2):
        lines += [f"### F = {F}", "", "| Method | No attack | Mean over attacks | Worst attack | Change vs LF (mean) | Wins vs LF | Safe? |",
                  "|---|---|---|---|---|---|---|"]
        lf = {a: res[("LF", F, a)]["error"].mean() for a in atts}
        lfm = np.mean([lf[a] for a in atts if a != "none"])
        for n in names:
            errs = {a: res[(n, F, a)]["error"].mean() for a in atts}
            att = {a: e for a, e in errs.items() if a != "none"}
            mean, worst = np.mean(list(att.values())), max(att, key=att.get)
            wins = sum(errs[a] < lf[a] for a in atts)
            flags = {bench.flag(res[(n, F, a)]) for a in atts} - {""}
            lines.append(f"| {n} | {errs['none']:.3f} | {mean:.3f} | {att[worst]:.3f} ({worst}) | "
                         f"{100 * (mean / lfm - 1):+.1f}% | {wins}/{len(atts)} | "
                         f"{'yes' if not flags else ', '.join(sorted(flags))} |")
        lines += ["", "Per attack (error):", "", "| Attack | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for a in atts:
            lines.append(f"| {a} | " + " | ".join(f"{res[(n, F, a)]['error'].mean():.3f}" for n in names) + " |")
        lines.append("")
    return lines


def training_plot(train):
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    for r, c in zip(train, ["#2a78d6", "#eb6834", "#7a4fd6"]):
        y = np.convolve(r["loss"], np.ones(20) / 20, mode="valid")
        ax.plot(y, color=c, lw=1.4, label=r["name"])
    ax.axhline(0, color="#888", lw=0.8, ls=":")
    ax.set(xlabel="training iteration", ylabel="(loss − LF loss) / LF loss\n(20-iteration moving average)",
           title="memory + fallback: training (below 0 = better than LF)")
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
    if (OUT / "training.json").exists():
        training_plot(json.loads((OUT / "training.json").read_text()))
    methods = {"LF": lambda: ts.LF(), "Su-Vaidya": lambda: ts.BASELINES["Su-Vaidya"]()}
    methods.update({n: (lambda n=n: load(n).to(args.device)) for n in VARIANTS})
    names = list(methods)
    out, md = {}, ["# memory + fallback defender", ""]
    for key, insts_for, title in (("unseen", lambda F: Lr.train_pool(F, UNSEEN, offset=300_000), f"{UNSEEN} unseen graphs"),
                                  ("benchmark", Lr.eval_instances, "benchmark (10 graphs)")):
        res = eval_on(methods, insts_for, args.device)
        out[key] = {f"{n}|F{F}|{a}": {k: v.tolist() for k, v in m.items()} for (n, F, a), m in res.items()}
        md += table(res, names, title, UNSEEN if key == "unseen" else 10)
    (OUT / "trust.json").write_text(json.dumps(out))
    (OUT / "trust.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
