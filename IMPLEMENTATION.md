# Implementation Log

Progress on the implementation and validation of the [research proposal](README.md). This file is updated at the end of every work session: what was built, what was verified, what we learned, and what comes next.

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Literature check | Partly done (see proposal §3); Scholar sweep still open |
| 1 | Simulator + reproduction of [1] | **Done** (2026-10-28) |
| 2 | Baselines (DGD, LF, Su–Vaidya, Q-consensus, DRC-RSSG) | DGD and LF done; others next |
| 3 | Scripted attack benchmark | Fixed, random, switching done; stealthy and trim-hugging next |
| 4 | RL attacker | Not started |
| 5 | Shielded defender + proof | Not started |
| 6 | Zero-sum training | Not started |
| 7 | Evaluation | Not started |
| 8 | Paper | Not started |

## Repository layout

```
advagent/
  graphs.py       example graphs of [1], random r-robust graphs, robustness checks,
                  maximum F-local set (integer program), F-local adversary sampling
  functions.py    local objectives: capped quadratics and |x - a|; x* and minimizer hull
  dynamics.py     LF filter, DGD and LF aggregation rules, step-size schedules, simulate()
  attacks.py      scripted adversaries: Fixed, UniformRandom (malicious/Byzantine), Switching
experiments/
  phase1_reproduce.py   regenerates results/phase1/
tests/
  test_phase1.py  15 regression tests, one per claim of [1] we reproduce
results/phase1/   figures + summary.json
```

## How to run

```bash
# tests (the env var stops ROS's globally installed pytest plugins from loading)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q

# Phase 1 figures and numbers
PYTHONPATH=. python3 experiments/phase1_reproduce.py
```

Requirements so far: Python 3.13, NumPy, SciPy, Matplotlib, NetworkX (tests only). PyTorch is not yet installed; it is needed from Phase 4.

## Simulator conventions

- `adj[i, j] = True` means edge (v_j → v_i): node i receives node j's value.
- One step for each regular node i: form the consensus point y_i from its own value and received values, then x_i ← y_i − α_t · d_i, with d_i a subgradient of f_i at y_i, exactly as in eq. (8) of [1].
- Adversaries return either one broadcast value each (malicious) or an n × |A| matrix of per-receiver values (Byzantine).
- **LF:** drops up to F values strictly above and F strictly below the node's own value, then averages uniformly over itself and the retained set (weight 1/(|J_i|+1) ≥ η).
- **DGD:** uses Metropolis weights 1/(1 + max(d_i, d_j)), which are doubly stochastic on undirected graphs.
- **Step sizes:** `power_steps(c, p)` gives α_t = c/(t+1)^p. Default c = 0.5, p = 0.6, which satisfies Σα = ∞ and Σα² < ∞.

## Phase 1 results: reproduction of [1]

All 15 tests pass. Numbers are from `results/phase1/summary.json`.

| Claim in [1] | Our result |
|---|---|
| Fig. 1 graph is 2-robust and (2,2)-robust, not 3-robust | Confirmed by brute force |
| Fig. 3 (U–W) graph is 3-robust; U is a maximum 1-local set | Confirmed (integer program, K = 2, 3, 5) |
| Maximum F-local set of a complete graph has size F | Confirmed for K5, K8 with F = 1, 2 |
| Prop. 4.3: one fixed adversary captures DGD | DGD mean reaches 42.9 of the adversary's 50 by t = 3000; LF stays at −0.15, inside the hull [−0.73, 0.58] |
| Ex. 6.2: LF-filtered graph not rooted, yet consensus under any single adversary | Kept edges match Fig. 1(b); filtered graph not rooted; spread < 1e−9 for all 5 adversary positions × 4 attacks, including Byzantine |
| Ex. 7.2: switching malicious node prevents convergence | 25 phase switches in 3000 steps; late mean keeps swinging between 0.25 and 2.00; no hull violation after t = 10 |
| Ex. 8.2 / Prop. 8.1: LF error equals b/4 on the U–W graph | Error 0.990 for b = 4 (b/4 = 1), identical for K = 2, 5, 10, 20; DGD error 1.5e−8 |
| Thms. 6.1 and 7.1: consensus + hull for F-local Byzantine on (2F+1)-robust graphs | Confirmed on 10 random 14-node graphs, F = 1, 2, random Byzantine noise in [−100, 100] |

Figures:

| | |
|---|---|
| ![Prop 4.3](results/phase1/prop4_3_capture.png) | ![Ex 7.2](results/phase1/ex7_2_oscillation.png) |
| ![Ex 8.2](results/phase1/ex8_2_lower_bound.png) | |

## Findings and decisions

These were not obvious from the paper and affect later phases.

1. **DGD needs doubly stochastic weights.** With uniform weights, DGD on the U–W graph converged to 0.64 instead of x* = 1, even with no adversaries. Lemma 3.4 of [1] explains why: the nodes minimize a centrality-weighted sum. We switched DGD to Metropolis weights, after which it is exact.
2. **Uniform LF weights cause the same bias, and this is where a learned defender can help.** LF averages uniformly over the retained set, so it too targets a centrality-weighted objective, not the true average. Learned weights inside the shield (Formulation B) can correct this without leaving the set where [1]'s theorems hold. This is concrete early evidence for RQ3. *To do in Phase 2:* add an "LF + Metropolis-style weights" baseline, to separate the gain from reweighting from the gain from learning.
3. **The maximum F-local set must be a proper subset.** S = V is vacuously F-local. We require at least one node outside S, which reproduces the paper's "size F in complete graphs".
4. **The Ex. 7.2 attack needs hysteresis.** A naive "switch when settled" rule chatters every step once nodes agree. The working attack switches on the regular mean: expose below 0.25, shadow above 2.0. These thresholds sit inside the two phase equilibria, which we measured at 0 and 2.45. An RL attacker (Phase 4) should rediscover this on its own.
5. **Harmonic steps kill the oscillation in practice.** With α_t = c/(t+1), each Ex. 7.2 cycle takes much longer than the last (only 2–6 switches in 5000 steps). With α_t = 0.5/(t+1)^0.6 the oscillation stays visible. Experiments should report the step-size schedule, since the attacker's success depends on it.
6. **Hull violations at t = 1 are expected.** A large first step (α_0 ≈ 1) can overshoot the hull. Theorem 7.1 is asymptotic, so safety metrics use a short burn-in (10 steps).

## Next steps

1. **Phase 2.** Implement Su–Vaidya [5], Q-consensus [7] + gradient step, and DRC-RSSG [10] + gradient step. DRC-RSSG needs a MILP per node per step; plan to use `scipy.optimize.milp`. Add the "LF + reweighting" baseline from finding 2.
2. **Phase 3.** Add the stealthy fake-cost attack (Thm. 4.4) and the trim-boundary hugging attack. Build the baseline × attack table.
3. Vectorise `simulate()` across episodes before RL training. The current per-node Python loop is fine for validation (~1 s per 3000-step run on 10 nodes) but too slow for training.
4. Install PyTorch (+ Gymnasium/PettingZoo) for Phase 4.

## Changelog

- **2026-10-28.** Phase 1 complete: simulator, example graphs, robustness and F-local tools, DGD/LF, three scripted attacks, 15 passing tests, three reproduction figures.
