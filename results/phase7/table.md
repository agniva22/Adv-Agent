# Phase 7: final evaluation

Phase 3 benchmark instances (10 random 20-node graphs per F, T = 2000). 'RL attacker (best response)' is an RL attacker trained against that specific method (Phase 4 for the baselines, here for the learned defenders); it picks the worse push direction per instance. Worst case and mean are over all attacks except 'none'. Unsafe: some run's hull violation (t ≥ T/2) or α = 0 range excess above 1e−3. No consensus: final spread above 1.

## Summary, F = 1

| Method | No attack | Mean over attacks | Worst case (attack) | Best-response RL attacker | Safe under every attack? |
|---|---|---|---|---|---|
| DGD | 0.000 | 8.594 | 39.499 (fixed +50) | 22.650 | **no**: unsafe under fixed +50, random Byzantine, stealthy (fake 50), RL attacker (best response) |
| LF | 0.137 | 0.286 | 0.539 (RL attacker (best response)) | 0.539 | yes |
| LF-Metropolis | 0.123 | 0.361 | 0.529 (RL attacker (best response)) | 0.529 | yes |
| Su-Vaidya | 0.237 | 0.377 | 0.645 (RL attacker (best response)) | 0.645 | yes |
| Trimmed mean | 0.141 | 0.275 | 0.517 (RL attacker (best response)) | 0.517 | yes |
| Q-consensus | 0.123 | 1.670 | 10.555 (RL attacker (best response)) | 10.555 | **no**: unsafe under fixed +50, random Byzantine, stealthy (fake 5), RL attacker (best response) |
| Q-consensus (fixed beta) | 0.138 | 0.572 | 1.819 (stealthy (fake 50)) | 1.361 | **no**: unsafe under fixed +50, random Byzantine, RL attacker (best response) |
| DRC-RSSG | 0.202 | 0.300 | 0.666 (RL attacker (best response)) | 0.666 | **no**: unsafe under farthest evader |
| Shielded RL | 0.137 | 0.284 | 0.540 (RL attacker (best response)) | 0.540 | yes |
| Shielded RL (game) | 0.136 | 0.283 | 0.538 (RL attacker (best response)) | 0.538 | yes |
| Shielded RL (no adv. training) | 0.118 | 0.307 | 0.528 (RL attacker (best response)) | 0.528 | yes |
| Unshielded RL | 0.137 | 5.729 | 39.028 (fixed +50) | 2.012 | **no**: unsafe under fixed +50, random Byzantine, stealthy (fake 50), RL attacker (best response) |

## Summary, F = 2

| Method | No attack | Mean over attacks | Worst case (attack) | Best-response RL attacker | Safe under every attack? |
|---|---|---|---|---|---|
| DGD | 0.000 | 7.795 | 44.701 (fixed +50) | — | **no**: unsafe under fixed +50, random Byzantine, stealthy (fake 50) |
| LF | 0.291 | 0.611 | 0.898 (RL attacker (best response)) | 0.898 | yes |
| LF-Metropolis | 0.242 | 0.653 | 0.897 (RL attacker (best response)) | 0.897 | yes |
| Su-Vaidya | 0.250 | 0.544 | 0.876 (RL attacker (best response)) | 0.876 | yes |
| Trimmed mean | 0.288 | 0.598 | 0.852 (RL attacker (best response)) | 0.852 | yes |
| Q-consensus | 0.162 | 0.517 | 1.509 (fixed +50) | — | **no**: unsafe under fixed +50, random Byzantine, stealthy (fake 5), stealthy (fake 50) |
| Q-consensus (fixed beta) | 0.106 | 0.637 | 2.274 (stealthy (fake 50)) | — | **no**: unsafe under fixed +50, random Byzantine, stealthy (fake 50) |
| DRC-RSSG | 0.402 | 0.568 | 1.144 (RL attacker (best response)) | 1.144 | **no**: unsafe under farthest evader |
| Shielded RL | 0.291 | 0.606 | 0.901 (RL attacker (best response)) | 0.901 | yes |
| Shielded RL (game) | 0.293 | 0.607 | 0.900 (RL attacker (best response)) | 0.900 | yes |
| Shielded RL (no adv. training) | 0.273 | 0.624 | 0.908 (RL attacker (best response)) | 0.908 | yes |
| Unshielded RL | 0.100 | 8.148 | 44.589 (fixed +50) | 12.318 | **no**: unsafe under fixed +50, random Byzantine, stealthy (fake 50), RL attacker (best response) |

## Full grid, F = 1: mean error

| Method | none | fixed +50 | random Byzantine | farthest evader | stealthy (fake 5) | stealthy (fake 50) | trim-hugger | switching | RL attacker (best response) |
|---|---|---|---|---|---|---|---|---|---|
| DGD | 0.000 | 39.499 ⚠ | 1.308 ⚠ | 0.500 | 0.387 | 3.917 ⚠ | 0.300 | 0.194 | 22.650 ⚠ |
| LF | 0.137 | 0.277 | 0.143 | 0.257 | 0.250 | 0.277 | 0.303 | 0.245 | 0.539 |
| LF-Metropolis | 0.123 | 0.397 | 0.100 | 0.370 | 0.289 | 0.397 | 0.384 | 0.421 | 0.529 |
| Su-Vaidya | 0.237 | 0.391 | 0.225 | 0.387 | 0.357 | 0.384 | 0.372 | 0.251 | 0.645 |
| Trimmed mean | 0.141 | 0.269 | 0.149 | 0.251 | 0.241 | 0.269 | 0.266 | 0.242 | 0.517 |
| Q-consensus | 0.123 ∅ | 1.281 ⚠ | 0.190 ⚠ | 0.205 ∅ | 0.204 ⚠ | 0.145 ∅ | 0.221 ∅ | 0.557 ∅ | 10.555 ⚠ |
| Q-consensus (fixed beta) | 0.138 | 0.134 ⚠ | 0.161 ⚠ | 0.337 | 0.255 | 1.819 | 0.225 | 0.282 | 1.361 ⚠ |
| DRC-RSSG | 0.202 | 0.140 | 0.149 | 0.491 ⚠ | 0.280 | 0.140 | 0.310 | 0.225 | 0.666 |
| Shielded RL | 0.137 | 0.276 | 0.144 | 0.254 | 0.248 | 0.276 | 0.302 | 0.234 | 0.540 |
| Shielded RL (game) | 0.136 | 0.276 | 0.142 | 0.258 | 0.249 | 0.276 | 0.305 | 0.221 | 0.538 |
| Shielded RL (no adv. training) | 0.118 | 0.303 | 0.134 | 0.272 | 0.250 | 0.303 | 0.312 | 0.357 | 0.528 |
| Unshielded RL | 0.137 | 39.028 ⚠ | 0.910 ⚠ | 0.337 | 0.242 | 2.877 ⚠ | 0.211 | 0.216 | 2.012 ⚠ |

## Full grid, F = 2: mean error

| Method | none | fixed +50 | random Byzantine | farthest evader | stealthy (fake 5) | stealthy (fake 50) | trim-hugger | switching | RL attacker (best response) |
|---|---|---|---|---|---|---|---|---|---|
| DGD | 0.000 | 44.701 ⚠ | 0.915 ⚠ | 1.000 | 0.655 | 6.632 ⚠ | 0.398 | 0.266 | — |
| LF | 0.291 | 0.692 | 0.216 | 0.679 | 0.658 | 0.692 | 0.666 | 0.390 | 0.898 |
| LF-Metropolis | 0.242 | 0.794 | 0.161 | 0.777 | 0.656 | 0.794 | 0.682 | 0.459 | 0.897 |
| Su-Vaidya | 0.250 | 0.676 | 0.131 | 0.669 | 0.530 | 0.533 | 0.540 | 0.398 | 0.876 |
| Trimmed mean | 0.288 | 0.685 | 0.214 | 0.667 | 0.651 | 0.685 | 0.618 | 0.412 | 0.852 |
| Q-consensus | 0.162 ∅ | 1.509 ⚠ | 0.286 ⚠ | 0.152 ∅ | 0.292 ⚠ | 0.185 ⚠ | 0.265 ∅ | 0.929 ∅ | — |
| Q-consensus (fixed beta) | 0.106 | 0.102 ⚠ | 0.177 ⚠ | 0.797 | 0.565 | 2.274 ⚠ | 0.365 | 0.181 | — |
| DRC-RSSG | 0.402 | 0.200 | 0.207 | 0.974 ⚠ | 0.679 | 0.207 | 0.701 | 0.432 | 1.144 |
| Shielded RL | 0.291 | 0.689 | 0.218 | 0.676 | 0.655 | 0.689 | 0.663 | 0.356 | 0.901 |
| Shielded RL (game) | 0.293 | 0.691 | 0.219 | 0.679 | 0.658 | 0.691 | 0.666 | 0.352 | 0.900 |
| Shielded RL (no adv. training) | 0.273 | 0.720 | 0.215 | 0.693 | 0.661 | 0.720 | 0.667 | 0.407 | 0.908 |
| Unshielded RL | 0.100 | 44.589 ⚠ | 0.683 ⚠ | 0.819 | 0.546 | 5.547 ⚠ | 0.348 | 0.331 | 12.318 ⚠ |

## U–W graph (Ex. 8.2, b = 4, no adversaries): error vs the |T|/n·b = b/4 = 1 bound

| Method | K = 2 | K = 5 |
|---|---|---|
| DGD | 0.000 | 0.000 |
| LF | 0.990 | 0.990 |
| LF-Metropolis | 0.976 | 0.947 |
| Su-Vaidya | 1.000 | 1.000 |
| Trimmed mean | 0.993 | 0.993 |
| Q-consensus | 0.788 | 0.910 |
| Q-consensus (fixed beta) | 0.371 | 0.696 |
| DRC-RSSG | 0.993 | 0.993 |
| Shielded RL | 0.990 | 0.990 |
| Shielded RL (game) | 0.990 | 0.990 |
| Shielded RL (no adv. training) | 0.989 | 0.988 |
| Unshielded RL | 0.414 | 0.718 |
