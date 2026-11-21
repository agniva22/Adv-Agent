# Learning to Aggregate Safely: Shielded Reinforcement Learning for Resilient Distributed Optimization

Code for the L4DC 2027 submission. Regular agents minimise the average of their local convex functions while up to $F$ adversaries in each neighbourhood send arbitrary values. We learn the aggregation weights of Local Filtering (LF) with reinforcement learning inside a **shield** (the LF filter followed by a floored softmax). Because of the shield, every learned policy keeps the consensus and safety guarantees of LF. A learned best-response attacker measures how much accuracy each method loses in the worst case.

## Setup

```bash
conda env create -f environment.yml      # Python 3.12, PyTorch, NumPy, SciPy, NetworkX
conda activate advagent
pytest                                   # about 3 minutes
```

To install with pip instead, run `pip install -e .`. The figure script also needs Ghostscript and `pdftops` (poppler-utils), which it uses to convert text to outlines.

## Reproducing the paper

The trained models and raw results are included, so every table can be checked without retraining. To rerun everything from scratch, run the scripts in this order. Each script reuses the outputs of the scripts above it. A GPU is optional.

| Paper | Script | Output |
|---|---|---|
| Scripted attack benchmark | `experiments/phase3_attacks.py` | `results/phase3/` |
| Table: attacker strength | `experiments/phase4_attacker.py` | `results/phase4/` |
| Shielded and unshielded RL (Table I, ablation) | `experiments/phase5_defender.py` | `results/phase5/` |
| Memory + fallback defender (Sec. 3) | `experiments/defender_trust.py` | `results/defender_trust/` |
| Table IV: zero-sum game | `experiments/phase6_game.py` | `results/phase6_trust/` |
| Tables I and II: safety and accuracy (100 unseen graphs) | `experiments/phase7_eval.py --unseen` | `results/phase7_unseen/` |
| Su–Vaidya with learned gradients | `experiments/sv_gradient_attacker.py` | `results/sv_gradient_attacker/` |
| Table III: seeds, ablation and Wilcoxon tests | `experiments/defender_seeds.py` | `results/defender_seeds/` |
| Larger networks ($n = 50, 100$) | `experiments/scale_test.py` | `results/scale_test/` |
| Modelling assumptions (Appendix) | `experiments/modeling_assumptions.py` | `results/modeling/` |
| Figures 1–3 | `experiments/paper_figures.py` | `results/paper_figures/` |

`phase7_eval.py` without `--unseen` evaluates on the 10 benchmark graphs (`results/phase7/`), and `--unseen` reuses the best-response attackers trained there. Example 1 (the DRC-RSSG counterexample) is checked by `tests/test_phase2.py::test_drc_rssg_safety_counterexample`.

## Repository layout

```
advagent/            library
  graphs.py          random r-robust graphs, F-local adversary sets
  dynamics.py        reference NumPy simulator (filtered subgradient dynamics)
  baselines.py       LF, LF-Metropolis, trimmed mean, Su–Vaidya, IOS, ClippedGossip, Q-consensus, DRC-RSSG
  attacks.py         scripted adversaries
  torchsim.py        batched, differentiable PyTorch simulator
  learn.py           RL attacker, shielded defender, memory + fallback defender, training loops
  bench.py           shared attack set and evaluation
experiments/         one script per result (table above)
results/             trained models (*.pt), raw results (*.json) and tables (*.md)
tests/               safety, shield and simulator tests
```

## Citation

```bibtex
@inproceedings{shielded2027,
  title     = {Learning to Aggregate Safely: Shielded Reinforcement Learning for Resilient Distributed Optimization},
  booktitle = {Learning for Dynamics and Control (L4DC)},
  year      = {2027}
}
```
