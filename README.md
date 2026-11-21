# Learning to Aggregate Safely: Shielded Reinforcement Learning for Resilient Distributed Optimization

Code for the L4DC 2027 submission.

Regular agents minimise the average of their local convex functions over a network in which up to $F$ neighbours of each agent are adversarial. Local Filtering (LF) is safe but gives up accuracy. We learn the aggregation weights of LF with reinforcement learning inside a **shield**: the LF filter followed by a floored softmax. Every policy in the shield keeps the consensus and safety guarantees of LF, whatever it has learned. Our defender (*memory + fallback*) adds a per-neighbour trust memory and a fallback to LF. A learned best-response attacker measures each method's worst-case error.

## Installation

```bash
git clone https://github.com/agniva22/Adv-Agent.git && cd Adv-Agent
conda env create -f environment.yml && conda activate advagent    # or: pip install -e .
pytest                                                            # about 3 minutes
```

The trained models and raw results are in `results/`, so nothing has to be retrained to check the paper.

```python
import torch
from advagent import bench, checkpoints as C, learn as Lr, torchsim as ts

graphs = Lr.train_pool(1, 10, offset=300_000)          # 10 of the unseen test graphs, F = 1
for name, rule in (("LF", ts.LF), ("memory + fallback", lambda: C.defender("memory + fallback"))):
    attacker = C.best_response_attacker(name, F=1)      # RL attacker trained against this method
    with torch.no_grad():
        m = bench.evaluate(rule, graphs, lambda: Lr.LearnedAttack(attacker, direc=1))
    print(name, m["error"].mean())
```

## Reproducing the paper

Each script `experiments/NN_<name>.py` writes `results/<name>/`: a `table.md` with the numbers, the raw `results.json`, and the trained models (`*.pt`). The scripts are numbered in the order they must run, because each one uses the models of the earlier ones. All of them train on CPU; a GPU is used if available.

| Script | Paper | What it does |
|---|---|---|
| `01_scripted_attacks.py` | benchmark | every method against the seven scripted attacks |
| `02_rl_attackers.py` | Table (attacker strength) | trains an RL attacker against each method |
| `03_shielded_rl.py` | Table I, ablation | shielded policy without memory, and the unshielded ablation |
| `04_memory_fallback.py` | Section 3 | trains the memory + fallback defender |
| `05_game.py` | Table IV | three rounds of the zero-sum game |
| `06_evaluate.py` | Tables I and II | all methods on 100 unseen graphs, each against its own best-response attacker |
| `07_su_vaidya_gradients.py` | Table (attacker strength) | attacker that also learns the fake gradients sent to Su–Vaidya |
| `08_seeds_ablation.py` | Table III, ablation | training seeds, memory-only and fallback-only variants, Wilcoxon tests |
| `09_larger_networks.py` | larger networks | $n = 50$ and $100$ without retraining |
| `10_modeling_assumptions.py` | Appendix | graph families, costs, link failures, asynchrony, wrong $F$ |
| `11_figures.py` | Figures 1–3 | simulations and figure panels (`--plot-only` redraws from `data.npz`) |

Run them from the repository root, e.g. `python experiments/06_evaluate.py`. `06_evaluate.py --benchmark` evaluates on the 10 benchmark graphs instead (`results/evaluation_benchmark/`). Example 1 (the DRC-RSSG counterexample) is the test `tests/test_baselines.py::test_drc_rssg_safety_counterexample`. The figure script also needs Ghostscript and `pdftops` (poppler-utils).

## Repository structure

```
advagent/                     library
├── graphs.py                 r-robust graphs, F-local adversary sets
├── functions.py              local cost functions
├── dynamics.py               reference NumPy simulator of the filtered subgradient dynamics
├── baselines.py              LF, LF-Metropolis, trimmed mean, Su–Vaidya, IOS, ClippedGossip, Q-consensus, DRC-RSSG
├── attacks.py                scripted adversaries (NumPy)
├── torchsim.py               batched, differentiable PyTorch simulator with the same rules and attacks
├── learn.py                  RL attacker, shielded defender, memory + fallback defender, training
├── bench.py                  shared attack set and evaluation metrics
├── checkpoints.py            result folders and loaders for every trained model
└── plots.py                  shared plots
experiments/                  one script per result, numbered in run order
results/                      one folder per script: table.md, results.json, trained models
tests/                        simulator, baselines, attacks, shield guarantees
```

## Citation

```bibtex
@inproceedings{shielded2027,
  title     = {Learning to Aggregate Safely: Shielded Reinforcement Learning for Resilient Distributed Optimization},
  booktitle = {Learning for Dynamics and Control (L4DC)},
  year      = {2027}
}
```
