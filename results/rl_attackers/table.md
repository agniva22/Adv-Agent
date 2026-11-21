# RL attacker

PPO, 300 iterations per attacker; evaluated on the benchmark graphs (T = 2000, 10 graphs per F). The attacker picks its push direction per instance. Hull violation is over t ≥ T/2; excess is the α = 0 safety run.

## Bias attacker: mean error |x̄(T) − x*|

| Method | F | No attack | Best scripted attack | Per-instance best of all scripted | **RL attacker** | RL / best scripted | Max hull violation | Max excess (α = 0) |
|---|---|---|---|---|---|---|---|---|
| DGD | 1 | 0.000 | 39.499 (fixed +50) | 39.499 | **22.650** | 0.57× | 39.298 | 45.299 |
| LF | 1 | 0.137 | 0.303 (trim-hugger) | 0.365 | **0.539** | 1.78× | 0.000 | 0.000 |
| LF-Metropolis | 1 | 0.123 | 0.421 (switching) | 0.501 | **0.529** | 1.26× | 0.000 | 0.000 |
| Q-consensus | 1 | 0.123 | 1.281 (fixed +50) | 1.350 | **10.555** | 8.24× | 44.436 | 0.000 |
| Q-consensus (fixed beta) | 1 | 0.138 | 1.819 (stealthy (fake 50)) | 1.819 | **1.361** | 0.75× | 0.000 | 0.074 |
| DRC-RSSG | 1 | 0.202 | 0.491 (farthest evader) | 0.539 | **0.666** | 1.35× | 0.000 | 0.000 |
| LF | 2 | 0.291 | 0.692 (fixed +50) | 0.705 | **0.898** | 1.30× | 0.000 | 0.000 |
| LF-Metropolis | 2 | 0.242 | 0.794 (fixed +50) | 0.823 | **0.897** | 1.13× | 0.000 | 0.000 |
| DRC-RSSG | 2 | 0.402 | 0.974 (farthest evader) | 1.043 | **1.144** | 1.17× | 0.000 | 0.000 |
| Trimmed mean | 1 | 0.141 | 0.269 (fixed +50) | 0.383 | **0.517** | 1.92× | 0.000 | 0.000 |
| Trimmed mean | 2 | 0.288 | 0.685 (fixed +50) | 0.711 | **0.852** | 1.24× | 0.000 | 0.000 |
| Su-Vaidya | 1 | 0.237 | 0.391 (fixed +50) | 0.531 | **0.645** | 1.65× | 0.000 | 0.000 |
| Su-Vaidya | 2 | 0.250 | 0.676 (fixed +50) | 0.700 | **0.876** | 1.30× | 0.000 | 0.000 |
| IOS | 1 | 0.351 | 0.561 (farthest evader) | 0.660 | **0.864** | 1.54× | 0.000 | 0.000 |
| ClippedGossip | 1 | 0.107 | 25.404 (fixed +50) | 25.404 | **0.617** | 0.02× | 0.000 | 0.000 |
| IOS | 2 | 0.698 | 1.248 (farthest evader) | 1.359 | **1.561** | 1.25× | 3.750 | 3.731 |
| ClippedGossip | 2 | 0.173 | 34.751 (fixed +50) | 34.751 | **1.002** | 0.03× | 0.000 | 0.000 |

## Oscillation attacker: total variation of x̄ over t ≥ T/2

| Defence | Graph | Scripted switching | **RL attacker** (best of deterministic / sampled) | RL attacker, deterministic only | Max hull violation |
|---|---|---|---|---|---|
| LF | random, F = 1 | 1.785 | **0.755** | 0.106 | 0.000 |
| LF | Ex. 7.2 (K5) | 11.319 | **20.175** | 0.106 | 0.000 |
