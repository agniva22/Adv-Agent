# Final evaluation

100 unseen random 20-node graphs per F (seeds disjoint from training and from the benchmark), T = 2000. 'RL attacker (best response)' is an RL attacker trained against that specific method (02_rl_attackers.py for the baselines, 06_evaluate.py for the learned defenders); it picks the worse push direction per instance. Worst case and mean are over all attacks except 'none'. Unsafe: some run's hull violation (t ≥ T/2) or α = 0 range excess above 1e−3. No consensus: final spread above 1.

## Summary, F = 1

| Method | No attack | Mean over attacks | Worst case (attack) | Best-response RL attacker | Safe under every attack? |
|---|---|---|---|---|---|
| DGD | 0.000 | 8.629 | 41.156 (fixed +50) | 21.056 | **no**: unsafe under fixed +50, random Byzantine, stealthy (fake 50), RL attacker (best response) |
| LF | 0.215 | 0.403 | 0.609 (RL attacker (best response)) | 0.609 | yes |
| LF-Metropolis | 0.157 | 0.418 | 0.578 (RL attacker (best response)) | 0.578 | yes |
| Su-Vaidya | 0.220 | 0.447 | 0.672 (RL attacker (best response)) | 0.672 | yes |
| Trimmed mean | 0.231 | 0.399 | 0.602 (RL attacker (best response)) | 0.602 | yes |
| IOS | 0.390 | 0.441 | 0.918 (RL attacker (best response)) | 0.918 | **no**: unsafe under farthest evader |
| ClippedGossip | 0.123 | 3.911 | 27.147 (fixed +50) | 0.699 | **no**: unsafe under fixed +50, random Byzantine, stealthy (fake 50) |
| Q-consensus | 0.172 | 1.666 | 9.352 (RL attacker (best response)) | 9.352 | **no**: unsafe under fixed +50, random Byzantine, stealthy (fake 5), RL attacker (best response) |
| Q-consensus (fixed beta) | 0.161 | 0.658 | 1.814 (stealthy (fake 50)) | 1.602 | **no**: unsafe under fixed +50, random Byzantine, RL attacker (best response) |
| DRC-RSSG | 0.265 | 0.418 | 0.751 (RL attacker (best response)) | 0.751 | **no**: unsafe under random Byzantine, farthest evader, RL attacker (best response) |
| Shielded RL | 0.216 | 0.401 | 0.608 (RL attacker (best response)) | 0.608 | yes |
| Unshielded RL | 0.153 | 6.123 | 40.950 (fixed +50) | 2.226 | **no**: unsafe under fixed +50, random Byzantine, stealthy (fake 50), RL attacker (best response) |
| memory + fallback | 0.182 | 0.378 | 0.582 (RL attacker (best response)) | 0.582 | yes |
| memory + fallback (game) | 0.172 | 0.412 | 0.569 (RL attacker (best response)) | 0.569 | yes |

## Summary, F = 2

| Method | No attack | Mean over attacks | Worst case (attack) | Best-response RL attacker | Safe under every attack? |
|---|---|---|---|---|---|
| DGD | 0.000 | 7.690 | 44.513 (fixed +50) | — | **no**: unsafe under fixed +50, random Byzantine, stealthy (fake 50) |
| LF | 0.277 | 0.599 | 0.889 (RL attacker (best response)) | 0.889 | yes |
| LF-Metropolis | 0.217 | 0.615 | 0.851 (RL attacker (best response)) | 0.851 | yes |
| Su-Vaidya | 0.249 | 0.600 | 0.956 (RL attacker (best response)) | 0.956 | yes |
| Trimmed mean | 0.289 | 0.587 | 0.871 (RL attacker (best response)) | 0.871 | yes |
| IOS | 0.634 | 0.662 | 1.457 (RL attacker (best response)) | 1.457 | **no**: unsafe under farthest evader, RL attacker (best response) |
| ClippedGossip | 0.181 | 4.875 | 33.556 (fixed +50) | 0.901 | **no**: unsafe under fixed +50, random Byzantine, stealthy (fake 50), RL attacker (best response) |
| Q-consensus | 0.217 | 0.564 | 1.544 (fixed +50) | — | **no**: unsafe under fixed +50, random Byzantine, stealthy (fake 5), stealthy (fake 50) |
| Q-consensus (fixed beta) | 0.150 | 0.665 | 2.344 (stealthy (fake 50)) | — | **no**: unsafe under fixed +50, random Byzantine, stealthy (fake 50) |
| DRC-RSSG | 0.375 | 0.586 | 1.209 (RL attacker (best response)) | 1.209 | **no**: unsafe under farthest evader, RL attacker (best response) |
| Shielded RL | 0.277 | 0.596 | 0.887 (RL attacker (best response)) | 0.887 | yes |
| Unshielded RL | 0.147 | 7.500 | 44.417 (fixed +50) | 7.519 | **no**: unsafe under fixed +50, random Byzantine, stealthy (fake 50), RL attacker (best response) |
| memory + fallback | 0.216 | 0.565 | 0.829 (RL attacker (best response)) | 0.829 | yes |
| memory + fallback (game) | 0.184 | 0.564 | 0.785 (RL attacker (best response)) | 0.785 | yes |

## Full grid, F = 1: mean error

| Method | none | fixed +50 | random Byzantine | farthest evader | stealthy (fake 5) | stealthy (fake 50) | trim-hugger | switching | RL attacker (best response) |
|---|---|---|---|---|---|---|---|---|---|
| DGD | 0.000 | 41.156 ⚠ | 1.140 ⚠ | 0.568 | 0.415 | 4.126 ⚠ | 0.335 | 0.235 | 21.056 ⚠ |
| LF | 0.215 | 0.426 | 0.187 | 0.400 | 0.388 | 0.426 | 0.451 | 0.334 | 0.609 |
| LF-Metropolis | 0.157 | 0.493 | 0.108 | 0.457 | 0.375 | 0.493 | 0.481 | 0.363 | 0.578 |
| Su-Vaidya | 0.220 | 0.506 | 0.222 | 0.499 | 0.424 | 0.432 | 0.469 | 0.352 | 0.672 |
| Trimmed mean | 0.231 | 0.432 | 0.205 | 0.408 | 0.393 | 0.432 | 0.429 | 0.294 | 0.602 |
| IOS | 0.390 | 0.181 | 0.180 | 0.675 ⚠ | 0.444 | 0.181 | 0.487 | 0.467 | 0.918 ∅ |
| ClippedGossip | 0.123 | 27.147 ⚠ | 0.514 ⚠ | 0.430 | 0.402 | 1.593 ⚠ | 0.287 | 0.217 | 0.699 |
| Q-consensus | 0.172 ∅ | 1.852 ⚠ | 0.225 ⚠ | 0.213 ∅ | 0.261 ⚠ | 0.182 ∅ | 0.251 ∅ | 0.992 ∅ | 9.352 ⚠ |
| Q-consensus (fixed beta) | 0.161 | 0.167 ⚠ | 0.173 ⚠ | 0.503 | 0.394 | 1.814 | 0.337 | 0.270 | 1.602 ⚠ |
| DRC-RSSG | 0.265 | 0.198 | 0.210 ⚠ | 0.720 ⚠ | 0.417 | 0.198 | 0.470 | 0.376 | 0.751 ⚠ |
| Shielded RL | 0.216 | 0.423 | 0.188 | 0.398 | 0.386 | 0.423 | 0.448 | 0.331 | 0.608 |
| Unshielded RL | 0.153 | 40.950 ⚠ | 0.954 ⚠ | 0.504 | 0.377 | 3.407 ⚠ | 0.318 | 0.253 | 2.226 ⚠ |
| memory + fallback | 0.182 | 0.402 | 0.173 | 0.378 | 0.360 | 0.409 | 0.419 | 0.305 | 0.582 |
| memory + fallback (game) | 0.172 | 0.465 | 0.168 | 0.446 | 0.401 | 0.477 | 0.462 | 0.307 | 0.569 |

## Full grid, F = 2: mean error

| Method | none | fixed +50 | random Byzantine | farthest evader | stealthy (fake 5) | stealthy (fake 50) | trim-hugger | switching | RL attacker (best response) |
|---|---|---|---|---|---|---|---|---|---|
| DGD | 0.000 | 44.513 ⚠ | 1.038 ⚠ | 0.896 | 0.606 | 6.144 ⚠ | 0.368 | 0.263 | — |
| LF | 0.277 | 0.643 | 0.239 | 0.629 | 0.613 | 0.643 | 0.617 | 0.517 | 0.889 |
| LF-Metropolis | 0.217 | 0.733 | 0.153 | 0.718 | 0.614 | 0.733 | 0.626 | 0.493 | 0.851 |
| Su-Vaidya | 0.249 | 0.716 | 0.219 | 0.711 | 0.590 | 0.597 | 0.573 | 0.437 | 0.956 |
| Trimmed mean | 0.289 | 0.642 | 0.259 | 0.626 | 0.612 | 0.642 | 0.586 | 0.457 | 0.871 |
| IOS | 0.634 | 0.233 | 0.226 | 0.987 ⚠ | 0.777 | 0.233 | 0.743 | 0.638 | 1.457 ⚠ |
| ClippedGossip | 0.181 | 33.556 ⚠ | 0.476 ⚠ | 0.648 | 0.636 | 2.167 ⚠ | 0.338 | 0.275 | 0.901 ⚠ |
| Q-consensus | 0.217 ∅ | 1.544 ⚠ | 0.291 ⚠ | 0.257 ∅ | 0.347 ⚠ | 0.226 ⚠ | 0.343 ∅ | 0.941 ∅ | — |
| Q-consensus (fixed beta) | 0.150 | 0.169 ⚠ | 0.157 ⚠ | 0.786 | 0.565 | 2.344 ⚠ | 0.374 | 0.258 | — |
| DRC-RSSG | 0.375 | 0.249 | 0.260 | 0.763 ⚠ | 0.665 | 0.250 | 0.648 | 0.639 | 1.209 ⚠ |
| Shielded RL | 0.277 | 0.642 | 0.238 | 0.627 | 0.612 | 0.642 | 0.616 | 0.507 | 0.887 |
| Unshielded RL | 0.147 | 44.417 ⚠ | 0.831 ⚠ | 0.803 | 0.546 | 5.267 ⚠ | 0.356 | 0.263 | 7.519 ⚠ |
| memory + fallback | 0.216 | 0.628 | 0.202 | 0.618 | 0.577 | 0.633 | 0.562 | 0.471 | 0.829 |
| memory + fallback (game) | 0.184 | 0.684 | 0.174 | 0.667 | 0.592 | 0.691 | 0.541 | 0.379 | 0.785 |

## U–W graph (Ex. 8.2, b = 4, no adversaries): error vs the |T|/n·b = b/4 = 1 bound

| Method | K = 2 | K = 5 |
|---|---|---|
| DGD | 0.000 | 0.000 |
| LF | 0.990 | 0.990 |
| LF-Metropolis | 0.976 | 0.947 |
| Su-Vaidya | 1.000 | 1.000 |
| Trimmed mean | 0.993 | 0.993 |
| IOS | 0.986 | 0.957 |
| ClippedGossip | 0.983 | 0.333 |
| Q-consensus | 0.788 | 0.910 |
| Q-consensus (fixed beta) | 0.371 | 0.696 |
| DRC-RSSG | 0.993 | 0.993 |
| Shielded RL | 0.990 | 0.990 |
| Unshielded RL | 0.414 | 0.718 |
| memory + fallback | 0.990 | 0.990 |
| memory + fallback (game) | 0.990 | 0.990 |
