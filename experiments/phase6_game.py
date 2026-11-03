"""Phase 6: zero-sum training of the shielded defender (proposal Formulation 5C).

Alternating best responses with a growing attacker population (a simple form
of PSRO [20]):

  round r:  A_r  = PPO best response to the current defender D_r (F = 1 and 2,
                   warm-started from the previous round's attacker);
            e_r  = damage A_r does to D_r on the benchmark instances
                   (the defender's exploitability);
            D_r+1 = D_r fine-tuned against the Phase 5 training mix plus every
                   attacker A_0..A_r.

D_0 is the Phase 5 "Shielded RL" defender. Writes results/phase6/{table.md,
results.json, exploitability.png, defenders/, attackers/}.
    PYTHONPATH=. python3 experiments/phase6_game.py [--rounds R]

With ``--defender trust`` the game is played with the memory + fallback
defender (``learn.TrustDefender``, D_0 from ``experiments/defender_trust.py``)
and exploitability is measured on 100 unseen graphs per F instead of the 10
benchmark graphs; results go to results/phase6_trust/.
"""

import argparse
import copy
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
OUT = ROOT / "results" / "phase6"
P5 = ROOT / "results" / "phase5"
P4 = ROOT / "results" / "phase4"
TRUST = ROOT / "results" / "defender_trust" / "defenders" / "memory___fallback.pt"


def phase5():
    import importlib.util
    spec = importlib.util.spec_from_file_location("p5", ROOT / "experiments" / "phase5_defender.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def defender_from(state, kind="shielded"):
    if kind == "trust":
        pol = Lr.TrustDefender(c=0.05, name="memory + fallback (game)")
    else:
        pol = Lr.ShieldedDefender(shield=True, name="Shielded RL (game)")
    pol.load_state_dict(state)
    return pol.eval()


def attacker_job(job):
    """Best response to a fixed defender; returns the policy state and its damage on the benchmark."""
    F, d_state, a_state, iters, seed, kind = job
    torch.set_num_threads(3)
    defender = defender_from(d_state, kind)
    pol = Lr.AttackerPolicy()
    if a_state is not None:
        pol.load_state_dict(a_state)
    pol, curve = Lr.train_attacker(lambda: defender, Lr.train_pool(F, 512), iters=iters, policy=pol, seed=seed)
    dmg = damage(defender_from(d_state, kind), pol, F, kind)
    return {"F": F, "state": pol.state_dict(), "curve": curve, **dmg}


@torch.no_grad()
def damage(rule, attacker, F, kind="shielded"):
    """Mean error (attacker picks the worse push direction per instance) and safety."""
    insts = Lr.train_pool(F, 100, offset=300_000) if kind == "trust" else Lr.eval_instances(F)
    runs = [bench.evaluate(lambda: rule, insts, lambda dr=dr: Lr.LearnedAttack(attacker, direc=dr)) for dr in (1, -1)]
    worst = np.maximum(runs[0]["error"], runs[1]["error"])
    return {"error": float(worst.mean()), "per_instance": worst.tolist(),
            "hull_violation": float(max(r["hull_violation"].max() for r in runs)),
            "excess": float(max(r["excess"].max() for r in runs))}


def load_policy(path):
    pol = Lr.AttackerPolicy()
    pol.load_state_dict(torch.load(path, map_location="cpu"))
    return pol.eval()


def main(rounds, a_iters, d_iters, device, kind="shielded"):
    p5 = phase5()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "defenders").mkdir(exist_ok=True)
    (OUT / "attackers").mkdir(exist_ok=True)
    d_state = torch.load(TRUST, map_location="cpu") if kind == "trust" else p5.load_defender("Shielded RL").state_dict()
    torch.save(d_state, OUT / "defenders" / "D0.pt")
    prev = {F: (torch.load(P4 / "attackers" / f"LF_F{F}_bias_random.pt") if (P4 / "attackers" / f"LF_F{F}_bias_random.pt").exists() else None)
            for F in (1, 2)}
    population = {1: [], 2: []}
    log = []
    for r in range(rounds + 1):
        if all((OUT / "attackers" / f"A{r}_F{F}.pt").exists() for F in (1, 2)):   # resume: reuse, re-measure only
            res = []
            for F in (1, 2):
                pol = load_policy(OUT / "attackers" / f"A{r}_F{F}.pt")
                res.append({"F": F, "state": pol.state_dict(), "curve": [], **damage(defender_from(d_state, kind), pol, F, kind)})
        else:
            with get_context("spawn").Pool(2) as pool:
                res = pool.map(attacker_job, [(F, d_state, prev[F], a_iters, r, kind) for F in (1, 2)])
        entry = {"round": r}
        for out in res:
            F = out["F"]
            torch.save(out["state"], OUT / "attackers" / f"A{r}_F{F}.pt")
            prev[F] = out["state"]
            pol = Lr.AttackerPolicy()
            pol.load_state_dict(out["state"])
            population[F].append(pol.eval())
            entry[f"F{F}"] = {k: out[k] for k in ("error", "hull_violation", "excess", "curve", "per_instance")}
        log.append(entry)
        print(f"round {r}: exploitability F=1 {entry['F1']['error']:.3f}  F=2 {entry['F2']['error']:.3f}", flush=True)
        if r == rounds:
            break
        # Defender update against the scripted mix plus the whole attacker population.
        defender = defender_from(d_state, kind)
        mix = p5.training_mix(True, {F: None for F in (1, 2)})
        for F, pols in population.items():
            for pol in pols:
                for dr in (+1, -1):
                    mix.append((2.0 / len(pols), lambda F_, p=pol, d_=dr, Fa=F:
                                Lr.LearnedAttack(copy.deepcopy(p), direc=d_) if F_ == Fa else ts.Band(0.99 * d_)))
        if (OUT / "defenders" / f"D{r + 1}.pt").exists():   # resume
            d_state, hist = torch.load(OUT / "defenders" / f"D{r + 1}.pt", map_location="cpu"), []
            log[-1]["defender_loss"] = hist
            continue
        defender, hist = Lr.train_defender(defender, {F: Lr.train_pool(F, 512) for F in (1, 2)}, mix,
                                           T=2000, iters=d_iters, device=device, lr=1e-3, seed=100 + r,
                                           loss_type="diff" if kind == "trust" else "ratio")
        d_state = {k: v.cpu() for k, v in defender.state_dict().items()}
        torch.save(d_state, OUT / "defenders" / f"D{r + 1}.pt")
        log[-1]["defender_loss"] = hist
    torch.save(d_state, OUT / "defenders" / "final.pt")
    (OUT / "results.json").write_text(json.dumps(log))
    return log


def lf_reference(kind="shielded"):
    """Damage of the best-response attacker against plain LF (on the same instances as ``damage``)."""
    if kind == "trust":
        res = json.loads((ROOT / "results" / "defender_br" / "br.json").read_text())["unseen"]
        return {F: float(np.mean(res[f"LF|F{F}|RL attacker (best response)"]["error"])) for F in (1, 2)}
    path = P4 / "results.json"
    if not path.exists():
        return {}
    res = json.loads(path.read_text())
    return {r["F"]: r["error"] for r in res if r["method"] == "LF" and r["objective"] == "bias" and r["graph"] == "random"}


def report(log, kind="shielded"):
    lf = lf_reference(kind)
    lines = ["# Phase 6: zero-sum training", "",
             "Exploitability = mean error the round's best-response RL attacker causes on "
             + ("100 unseen graphs per F" if kind == "trust" else "the Phase 3 benchmark instances")
             + " (it picks the worse push direction per instance). D_0 is the "
             + ("memory + fallback defender" if kind == "trust" else "Phase 5 shielded defender")
             + "; each later D_r is trained against the scripted mix plus all attackers so far.", "",
             "| Round r | Defender | Exploitability, F = 1 | Exploitability, F = 2 | Max hull violation | Max excess (α = 0) |",
             "|---|---|---|---|---|---|"]
    for e in log:
        hv = max(e["F1"]["hull_violation"], e["F2"]["hull_violation"])
        ex = max(e["F1"]["excess"], e["F2"]["excess"])
        lines.append(f"| {e['round']} | D_{e['round']} | {e['F1']['error']:.3f} | {e['F2']['error']:.3f} | {hv:.3f} | {ex:.3f} |")
    if lf:
        lines.append(f"| — | plain LF (Phase 4 attacker) | {lf.get(1, float('nan')):.3f} | {lf.get(2, float('nan')):.3f} | — | — |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    rr = [e["round"] for e in log]
    for F, c in ((1, "#2a78d6"), (2, "#eb6834")):
        ax.plot(rr, [e[f"F{F}"]["error"] for e in log], "o-", color=c, lw=1.6, ms=6,
                label=f"{'memory + fallback' if kind == 'trust' else 'shielded RL'} defender, F = {F}")
        if F in lf:
            ax.axhline(lf[F], color=c, lw=1, ls=":", label=f"plain LF vs its RL attacker, F = {F}")
    ax.set(xlabel="game round r", ylabel="exploitability\n(error under best-response attacker)",
           title="Phase 6: exploitability over game rounds", xticks=rr)
    ax.legend(fontsize=7, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "exploitability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--a-iters", type=int, default=150)
    ap.add_argument("--d-iters", type=int, default=150)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--defender", choices=("shielded", "trust"), default="shielded")
    args = ap.parse_args()
    if args.defender == "trust":
        OUT = ROOT / "results" / "phase6_trust"
    log = json.loads((OUT / "results.json").read_text()) if args.report_only else main(args.rounds, args.a_iters, args.d_iters, args.device, args.defender)
    print(report(log, args.defender))
