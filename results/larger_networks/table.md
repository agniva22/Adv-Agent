# Bigger graphs

Mean error over 50 random graphs per (n, F), T = 2000. All policies were trained on 20 nodes only. Mean and worst are over all attacks except 'none', including 'RL attacker (best response)' (the method's own).

## n = 50, F = 1

| Method | No attack | Mean over attacks | Worst attack | Own best response | Safe? |
|---|---|---|---|---|---|
| LF | 0.164 | 0.316 | 0.523 (RL attacker (best response)) | 0.523 | yes |
| LF-Metropolis | 0.081 | 0.387 | 0.529 (RL attacker (best response)) | 0.529 | yes |
| Su-Vaidya | 0.196 | 0.363 | 0.597 (RL attacker (best response)) | 0.597 | yes |
| Trimmed mean | 0.171 | 0.314 | 0.520 (RL attacker (best response)) | 0.520 | yes |
| memory + fallback | 0.141 | 0.312 | 0.505 (RL attacker (best response)) | 0.505 | yes |
| memory + fallback (game) | 0.121 | 0.358 | 0.492 (RL attacker (best response)) | 0.492 | yes |

## n = 50, F = 2

| Method | No attack | Mean over attacks | Worst attack | Own best response | Safe? |
|---|---|---|---|---|---|
| LF | 0.175 | 0.498 | 0.722 (RL attacker (best response)) | 0.722 | yes |
| LF-Metropolis | 0.129 | 0.524 | 0.719 (RL attacker (best response)) | 0.719 | yes |
| Su-Vaidya | 0.159 | 0.518 | 0.779 (RL attacker (best response)) | 0.779 | yes |
| Trimmed mean | 0.181 | 0.485 | 0.715 (RL attacker (best response)) | 0.715 | yes |
| memory + fallback | 0.149 | 0.476 | 0.687 (RL attacker (best response)) | 0.687 | yes |
| memory + fallback (game) | 0.130 | 0.486 | 0.661 (RL attacker (best response)) | 0.661 | yes |

## n = 100, F = 1

| Method | No attack | Mean over attacks | Worst attack | Own best response | Safe? |
|---|---|---|---|---|---|
| LF | 0.128 | 0.349 | 0.507 (RL attacker (best response)) | 0.507 | yes |
| LF-Metropolis | 0.067 | 0.398 | 0.509 (RL attacker (best response)) | 0.509 | yes |
| Su-Vaidya | 0.128 | 0.366 | 0.535 (RL attacker (best response)) | 0.535 | yes |
| Trimmed mean | 0.130 | 0.343 | 0.495 (RL attacker (best response)) | 0.495 | yes |
| memory + fallback | 0.099 | 0.329 | 0.468 (RL attacker (best response)) | 0.468 | yes |
| memory + fallback (game) | 0.092 | 0.367 | 0.462 (RL attacker (best response)) | 0.462 | yes |

## n = 100, F = 2

| Method | No attack | Mean over attacks | Worst attack | Own best response | Safe? |
|---|---|---|---|---|---|
| LF | 0.116 | 0.453 | 0.634 (RL attacker (best response)) | 0.634 | yes |
| LF-Metropolis | 0.086 | 0.527 | 0.695 (RL attacker (best response)) | 0.695 | yes |
| Su-Vaidya | 0.123 | 0.503 | 0.728 (RL attacker (best response)) | 0.728 | yes |
| Trimmed mean | 0.123 | 0.449 | 0.621 (RL attacker (best response)) | 0.621 | yes |
| memory + fallback | 0.096 | 0.457 | 0.619 (RL attacker (best response)) | 0.619 | yes |
| memory + fallback (game) | 0.092 | 0.480 | 0.629 (RL attacker (best response)) | 0.629 | yes |

