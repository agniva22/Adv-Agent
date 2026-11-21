"""Where every experiment stores its outputs, and how to load the trained models.

Each script ``experiments/NN_<name>.py`` writes to ``results/<name>/``. Trained
models are PyTorch state dicts named ``slug(method)``, e.g. ``lf_metropolis``.
"""

import re
from pathlib import Path

import torch

from . import learn as Lr

RESULTS = Path(__file__).resolve().parents[1] / "results"
SCRIPTED_ATTACKS = RESULTS / "scripted_attacks"
RL_ATTACKERS = RESULTS / "rl_attackers"
SHIELDED_RL = RESULTS / "shielded_rl"
MEMORY_FALLBACK = RESULTS / "memory_fallback"
GAME = RESULTS / "game"
EVALUATION = RESULTS / "evaluation"                       # 100 unseen graphs per F (the paper)
EVALUATION_BENCHMARK = RESULTS / "evaluation_benchmark"   # the 10 benchmark graphs per F
SU_VAIDYA_GRADIENTS = RESULTS / "su_vaidya_gradients"
SEEDS_ABLATION = RESULTS / "seeds_ablation"
LARGER_NETWORKS = RESULTS / "larger_networks"
MODELING_ASSUMPTIONS = RESULTS / "modeling_assumptions"
FIGURES = RESULTS / "figures"

BR = "RL attacker (best response)"
BASELINES = ["DGD", "LF", "LF-Metropolis", "Su-Vaidya", "Trimmed mean", "IOS", "ClippedGossip",
             "Q-consensus", "Q-consensus (fixed beta)", "DRC-RSSG"]
OURS = ["memory + fallback", "memory + fallback (game)"]
# Defenders of experiments/03_shielded_rl.py: the shielded policy without memory, and its ablations.
SHIELDED_VARIANTS = {
    "Shielded RL": dict(shield=True, adversarial=True),
    "Shielded RL (no adv. training)": dict(shield=True, adversarial=False),
    "Unshielded RL": dict(shield=False, adversarial=True),
}


def slug(name):
    """File name of a method: 'memory + fallback (game)' -> 'memory_fallback_game'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load(pol, path):
    pol.load_state_dict(torch.load(path, map_location="cpu"))
    return pol.eval()


def rl_attacker_path(method, F, objective="bias", graph="random"):
    return RL_ATTACKERS / "attackers" / f"{slug(method)}_F{F}_{objective}_{graph}.pt"


def rl_attacker(method="LF", F=1, objective="bias", graph="random"):
    """The RL attacker trained against ``method`` (experiments/02_rl_attackers.py), or None."""
    path = rl_attacker_path(method, F, objective, graph)
    return _load(Lr.AttackerPolicy(), path) if path.exists() else None


def defender(name):
    """A trained defender by its name in the paper's tables."""
    if name == "memory + fallback":
        return _load(Lr.TrustDefender(c=0.05, name=name), MEMORY_FALLBACK / "defenders" / "memory_fallback.pt")
    if name == "memory + fallback (game)":
        return _load(Lr.TrustDefender(c=0.05, name=name), GAME / "defenders" / "final.pt")
    if name in SHIELDED_VARIANTS:
        pol = Lr.ShieldedDefender(shield=SHIELDED_VARIANTS[name]["shield"], name=name)
        return _load(pol, SHIELDED_RL / "defenders" / f"{slug(name)}.pt")
    raise KeyError(name)


def best_response_path(name, F, search=(EVALUATION, EVALUATION_BENCHMARK)):
    """Where the best-response attacker of a learned defender is stored (first match in ``search``)."""
    for d in search:
        if (d / "attackers" / f"{slug(name)}_F{F}.pt").exists():
            return d / "attackers" / f"{slug(name)}_F{F}.pt"
    return search[0] / "attackers" / f"{slug(name)}_F{F}.pt"


def best_response_attacker(name, F, search=(EVALUATION, EVALUATION_BENCHMARK)):
    """The RL attacker trained against ``name`` itself, or None if it has not been trained.

    For the baselines this is the attacker of experiments/02_rl_attackers.py; for Su-Vaidya,
    the attacker that also learns its fake gradients (experiments/07_su_vaidya_gradients.py).
    """
    if name == "Su-Vaidya" and (SU_VAIDYA_GRADIENTS / "attackers" / f"su_vaidya_F{F}.pt").exists():
        return _load(Lr.GradientAttacker(), SU_VAIDYA_GRADIENTS / "attackers" / f"su_vaidya_F{F}.pt")
    if name in BASELINES:
        return rl_attacker(name, F)
    path = best_response_path(name, F, search)
    return _load(Lr.AttackerPolicy(), path) if path.exists() else None
