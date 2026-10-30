# Phase 3 scripted-attack benchmark

10 seeds per F; random 20-node (2F+1)-robust graphs; up to 3F adversaries (F-local); T = 2000; alpha_t = 0.5/(t+1)^0.6. Unsafe = hull violation or excess over the initial range above 0.001; no consensus = final spread above 1.0.

## F = 1

### Error |x̄ − x*| (mean ± std over seeds); ⚠ = some run unsafe, ∅ = some run without consensus

| Method | none | fixed +50 | random Byzantine | farthest evader | stealthy (fake 5) | stealthy (fake 50) | trim-hugger | switching |
|---|---|---|---|---|---|---|---|---|
| DGD | 0.000 | 39.499 ⚠ | 1.231 ⚠ | 0.500 | 0.387 | 3.917 ⚠ | 0.300 | 0.194 |
| LF | 0.137 | 0.277 | 0.147 | 0.257 | 0.250 | 0.277 | 0.303 | 0.245 |
| LF-Metropolis | 0.123 | 0.397 | 0.097 | 0.370 | 0.289 | 0.397 | 0.384 | 0.421 |
| Su-Vaidya | 0.141 | 0.269 | 0.154 | 0.251 | 0.241 | 0.269 | 0.266 | 0.242 |
| Q-consensus | 0.123 ∅ | 1.281 ⚠ | 0.226 ⚠ | 0.205 ∅ | 0.204 ⚠ | 0.145 ∅ | 0.221 ∅ | 0.557 ∅ |
| Q-consensus (fixed beta) | 0.138 | 0.134 ⚠ | 0.203 ⚠ | 0.337 | 0.255 | 1.819 | 0.225 | 0.282 |
| DRC-RSSG | 0.202 | 0.140 | 0.150 | 0.491 ⚠ | 0.280 | 0.140 | 0.310 | 0.225 |

### F = 1, attack: none

| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |
|---|---|---|---|---|---|
| DGD | 0.000 ± 0.000 | 0.205 ± 0.037 | 0.000 ± 0.000 | 0.000 | 0.000 |
| LF | 0.137 ± 0.114 | 0.155 ± 0.025 | 0.001 ± 0.001 | 0.000 | 0.000 |
| LF-Metropolis | 0.123 ± 0.087 | 0.311 ± 0.051 | 0.004 ± 0.007 | 0.000 | 0.000 |
| Su-Vaidya | 0.141 ± 0.111 | 0.131 ± 0.022 | 0.001 ± 0.001 | 0.000 | 0.000 |
| Q-consensus | 0.123 ± 0.139 | 4.637 ± 2.138 | 0.399 ± 0.273 | 0.000 | 0.000 |
| Q-consensus (fixed beta) | 0.138 ± 0.094 | 0.130 ± 0.021 | 0.004 ± 0.002 | 0.000 | 0.000 |
| DRC-RSSG | 0.202 ± 0.116 | 0.146 ± 0.026 | 0.002 ± 0.001 | 0.000 | 0.000 |

### F = 1, attack: fixed +50

| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |
|---|---|---|---|---|---|
| DGD | 39.499 ± 3.554 | 3.554 ± 1.138 | 3.788 ± 0.695 | 40.893 | 46.499 |
| LF | 0.277 ± 0.239 | 0.154 ± 0.035 | 0.003 ± 0.003 | 0.000 | 0.000 |
| LF-Metropolis | 0.397 ± 0.251 | 0.323 ± 0.072 | 0.007 ± 0.007 | 0.000 | 0.000 |
| Su-Vaidya | 0.269 ± 0.246 | 0.135 ± 0.035 | 0.002 ± 0.002 | 0.000 | 0.000 |
| Q-consensus | 1.281 ± 0.723 | 12.348 ± 3.988 | 0.466 ± 0.248 | 12.712 | 46.337 |
| Q-consensus (fixed beta) | 0.134 ± 0.135 | 0.128 ± 0.025 | 0.098 ± 0.166 | 4.161 | 46.337 |
| DRC-RSSG | 0.140 ± 0.157 | 0.133 ± 0.028 | 0.001 ± 0.001 | 0.000 | 0.000 |

### F = 1, attack: random Byzantine

| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |
|---|---|---|---|---|---|
| DGD | 1.231 ± 0.888 | 9.412 ± 2.370 | 411.620 ± 86.002 | 13.548 | 11.705 |
| LF | 0.147 ± 0.122 | 0.154 ± 0.027 | 1.359 ± 0.477 | 0.000 | 0.000 |
| LF-Metropolis | 0.097 ± 0.052 | 0.320 ± 0.056 | 1.619 ± 0.572 | 0.000 | 0.000 |
| Su-Vaidya | 0.154 ± 0.144 | 0.134 ± 0.023 | 1.498 ± 0.524 | 0.000 | 0.000 |
| Q-consensus | 0.226 ± 0.178 | 3.975 ± 2.095 | 6.813 ± 5.241 | 0.000 | 9.030 |
| Q-consensus (fixed beta) | 0.203 ± 0.197 | 0.339 ± 0.111 | 13.982 ± 3.828 | 0.000 | 8.940 |
| DRC-RSSG | 0.150 ± 0.152 | 0.134 ± 0.028 | 0.028 ± 0.017 | 0.000 | 0.000 |

### F = 1, attack: farthest evader

| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |
|---|---|---|---|---|---|
| DGD | 0.500 ± 0.219 | 0.243 ± 0.067 | 0.008 ± 0.006 | 0.000 | 0.000 |
| LF | 0.257 ± 0.242 | 0.155 ± 0.036 | 0.003 ± 0.003 | 0.000 | 0.000 |
| LF-Metropolis | 0.370 ± 0.209 | 0.325 ± 0.082 | 0.045 ± 0.077 | 0.000 | 0.000 |
| Su-Vaidya | 0.251 ± 0.245 | 0.136 ± 0.036 | 0.002 ± 0.002 | 0.000 | 0.000 |
| Q-consensus | 0.205 ± 0.142 | 4.450 ± 2.103 | 0.317 ± 0.234 | 0.000 | 0.000 |
| Q-consensus (fixed beta) | 0.337 ± 0.251 | 0.151 ± 0.045 | 0.012 ± 0.011 | 0.000 | 0.000 |
| DRC-RSSG | 0.491 ± 0.356 | 0.186 ± 0.053 | 0.006 ± 0.004 | 0.000 | 0.628 |

### F = 1, attack: stealthy (fake 5)

| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |
|---|---|---|---|---|---|
| DGD | 0.387 ± 0.148 | 0.201 ± 0.038 | 0.004 ± 0.002 | 0.000 | 0.000 |
| LF | 0.250 ± 0.195 | 0.152 ± 0.032 | 0.002 ± 0.002 | 0.000 | 0.000 |
| LF-Metropolis | 0.289 ± 0.128 | 0.312 ± 0.054 | 0.004 ± 0.002 | 0.000 | 0.000 |
| Su-Vaidya | 0.241 ± 0.204 | 0.131 ± 0.028 | 0.002 ± 0.002 | 0.000 | 0.000 |
| Q-consensus | 0.204 ± 0.147 | 5.410 ± 2.451 | 0.365 ± 0.184 | 0.675 | 0.000 |
| Q-consensus (fixed beta) | 0.255 ± 0.168 | 0.127 ± 0.024 | 0.006 ± 0.005 | 0.000 | 0.000 |
| DRC-RSSG | 0.280 ± 0.235 | 0.146 ± 0.035 | 0.003 ± 0.002 | 0.000 | 0.000 |

### F = 1, attack: stealthy (fake 50)

| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |
|---|---|---|---|---|---|
| DGD | 3.917 ± 1.213 | 0.402 ± 0.093 | 0.039 ± 0.014 | 1.706 | 0.000 |
| LF | 0.277 ± 0.239 | 0.154 ± 0.035 | 0.003 ± 0.003 | 0.000 | 0.000 |
| LF-Metropolis | 0.397 ± 0.251 | 0.323 ± 0.072 | 0.007 ± 0.007 | 0.000 | 0.000 |
| Su-Vaidya | 0.269 ± 0.246 | 0.135 ± 0.035 | 0.002 ± 0.002 | 0.000 | 0.000 |
| Q-consensus | 0.145 ± 0.150 | 4.522 ± 2.088 | 0.346 ± 0.236 | 0.000 | 0.000 |
| Q-consensus (fixed beta) | 1.819 ± 0.668 | 0.193 ± 0.047 | 0.584 ± 0.251 | 0.000 | 0.000 |
| DRC-RSSG | 0.140 ± 0.157 | 0.133 ± 0.028 | 0.001 ± 0.001 | 0.000 | 0.000 |

### F = 1, attack: trim-hugger

| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |
|---|---|---|---|---|---|
| DGD | 0.300 ± 0.130 | 0.211 ± 0.047 | 0.004 ± 0.002 | 0.000 | 0.000 |
| LF | 0.303 ± 0.244 | 0.163 ± 0.039 | 0.004 ± 0.004 | 0.000 | 0.000 |
| LF-Metropolis | 0.384 ± 0.177 | 0.319 ± 0.074 | 0.006 ± 0.005 | 0.000 | 0.000 |
| Su-Vaidya | 0.266 ± 0.245 | 0.135 ± 0.035 | 0.002 ± 0.002 | 0.000 | 0.000 |
| Q-consensus | 0.221 ± 0.139 | 5.366 ± 2.416 | 0.317 ± 0.189 | 0.000 | 0.000 |
| Q-consensus (fixed beta) | 0.225 ± 0.170 | 0.131 ± 0.031 | 0.006 ± 0.002 | 0.000 | 0.000 |
| DRC-RSSG | 0.310 ± 0.323 | 0.151 ± 0.045 | 0.004 ± 0.004 | 0.000 | 0.000 |

### F = 1, attack: switching

| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |
|---|---|---|---|---|---|
| DGD | 0.194 ± 0.134 | 0.209 ± 0.048 | 1.430 ± 0.445 | 0.000 | 0.000 |
| LF | 0.245 ± 0.194 | 0.168 ± 0.038 | 1.785 ± 0.701 | 0.000 | 0.000 |
| LF-Metropolis | 0.421 ± 0.265 | 0.330 ± 0.061 | 1.896 ± 0.736 | 0.000 | 0.000 |
| Su-Vaidya | 0.242 ± 0.230 | 0.138 ± 0.033 | 1.729 ± 0.654 | 0.000 | 0.000 |
| Q-consensus | 0.557 ± 0.422 | 3.234 ± 2.416 | 6.053 ± 3.047 | 0.000 | 0.000 |
| Q-consensus (fixed beta) | 0.282 ± 0.158 | 0.131 ± 0.024 | 1.282 ± 0.467 | 0.000 | 0.000 |
| DRC-RSSG | 0.225 ± 0.228 | 0.156 ± 0.042 | 1.756 ± 0.711 | 0.000 | 0.000 |

## F = 2

### Error |x̄ − x*| (mean ± std over seeds); ⚠ = some run unsafe, ∅ = some run without consensus

| Method | none | fixed +50 | random Byzantine | farthest evader | stealthy (fake 5) | stealthy (fake 50) | trim-hugger | switching |
|---|---|---|---|---|---|---|---|---|
| DGD | 0.000 | 44.701 ⚠ | 1.408 ⚠ | 1.000 | 0.655 | 6.632 ⚠ | 0.398 | 0.266 |
| LF | 0.291 | 0.692 | 0.213 | 0.679 | 0.658 | 0.692 | 0.666 | 0.390 |
| LF-Metropolis | 0.242 | 0.794 | 0.162 | 0.777 | 0.656 | 0.794 | 0.682 | 0.459 |
| Su-Vaidya | 0.288 | 0.685 | 0.211 | 0.667 | 0.651 | 0.685 | 0.618 | 0.412 |
| Q-consensus | 0.162 ∅ | 1.509 ⚠ | 0.194 ⚠ | 0.152 ∅ | 0.292 ⚠ | 0.185 ⚠ | 0.265 ∅ | 0.929 ∅ |
| Q-consensus (fixed beta) | 0.106 | 0.102 ⚠ | 0.189 ⚠ | 0.797 | 0.565 | 2.274 ⚠ | 0.365 | 0.181 |
| DRC-RSSG | 0.402 | 0.200 | 0.210 | 0.974 ⚠ | 0.640 | 0.200 | 0.701 | 0.432 |

### F = 2, attack: none

| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |
|---|---|---|---|---|---|
| DGD | 0.000 ± 0.000 | 0.162 ± 0.026 | 0.000 ± 0.000 | 0.000 | 0.000 |
| LF | 0.291 ± 0.192 | 0.152 ± 0.027 | 0.003 ± 0.002 | 0.000 | 0.000 |
| LF-Metropolis | 0.242 ± 0.167 | 0.310 ± 0.060 | 0.006 ± 0.003 | 0.000 | 0.000 |
| Su-Vaidya | 0.288 ± 0.190 | 0.121 ± 0.023 | 0.002 ± 0.001 | 0.000 | 0.000 |
| Q-consensus | 0.162 ± 0.143 | 4.887 ± 1.205 | 0.427 ± 0.269 | 0.000 | 0.000 |
| Q-consensus (fixed beta) | 0.106 ± 0.064 | 0.114 ± 0.015 | 0.006 ± 0.003 | 0.000 | 0.000 |
| DRC-RSSG | 0.402 ± 0.229 | 0.154 ± 0.035 | 0.004 ± 0.002 | 0.000 | 0.000 |

### F = 2, attack: fixed +50

| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |
|---|---|---|---|---|---|
| DGD | 44.701 ± 1.028 | 1.927 ± 0.493 | 2.315 ± 0.182 | 42.704 | 46.161 |
| LF | 0.692 ± 0.247 | 0.136 ± 0.028 | 0.007 ± 0.005 | 0.000 | 0.000 |
| LF-Metropolis | 0.794 ± 0.228 | 0.297 ± 0.122 | 0.032 ± 0.059 | 0.000 | 0.000 |
| Su-Vaidya | 0.685 ± 0.250 | 0.116 ± 0.026 | 0.004 ± 0.002 | 0.000 | 0.000 |
| Q-consensus | 1.509 ± 1.340 | 13.797 ± 5.723 | 0.622 ± 0.273 | 14.609 | 46.161 |
| Q-consensus (fixed beta) | 0.102 ± 0.078 | 0.108 ± 0.020 | 0.491 ± 0.616 | 11.489 | 46.161 |
| DRC-RSSG | 0.200 ± 0.111 | 0.156 ± 0.076 | 0.001 ± 0.001 | 0.000 | 0.000 |

### F = 2, attack: random Byzantine

| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |
|---|---|---|---|---|---|
| DGD | 1.408 ± 0.724 | 9.886 ± 2.253 | 502.760 ± 53.762 | 12.912 | 11.843 |
| LF | 0.213 ± 0.102 | 0.137 ± 0.025 | 1.563 ± 0.258 | 0.000 | 0.000 |
| LF-Metropolis | 0.162 ± 0.081 | 0.288 ± 0.087 | 1.834 ± 0.452 | 0.000 | 0.000 |
| Su-Vaidya | 0.211 ± 0.103 | 0.116 ± 0.022 | 1.810 ± 0.431 | 0.000 | 0.000 |
| Q-consensus | 0.194 ± 0.143 | 3.341 ± 2.090 | 9.442 ± 10.458 | 1.396 | 9.509 |
| Q-consensus (fixed beta) | 0.189 ± 0.058 | 0.367 ± 0.102 | 16.805 ± 2.421 | 0.000 | 9.414 |
| DRC-RSSG | 0.210 ± 0.110 | 0.156 ± 0.076 | 0.069 ± 0.040 | 0.000 | 0.000 |

### F = 2, attack: farthest evader

| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |
|---|---|---|---|---|---|
| DGD | 1.000 ± 0.404 | 0.225 ± 0.145 | 0.018 ± 0.022 | 0.000 | 0.000 |
| LF | 0.679 ± 0.246 | 0.138 ± 0.028 | 0.005 ± 0.002 | 0.000 | 0.000 |
| LF-Metropolis | 0.777 ± 0.226 | 0.292 ± 0.114 | 0.031 ± 0.041 | 0.000 | 0.000 |
| Su-Vaidya | 0.667 ± 0.253 | 0.117 ± 0.026 | 0.004 ± 0.002 | 0.000 | 0.000 |
| Q-consensus | 0.152 ± 0.157 | 4.659 ± 1.793 | 0.491 ± 0.216 | 0.000 | 0.000 |
| Q-consensus (fixed beta) | 0.797 ± 0.223 | 0.143 ± 0.054 | 0.020 ± 0.016 | 0.000 | 0.000 |
| DRC-RSSG | 0.974 ± 0.405 | 0.399 ± 0.596 | 0.105 ± 0.169 | 0.000 | 1.457 |

### F = 2, attack: stealthy (fake 5)

| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |
|---|---|---|---|---|---|
| DGD | 0.655 ± 0.218 | 0.144 ± 0.037 | 0.005 ± 0.003 | 0.000 | 0.000 |
| LF | 0.658 ± 0.225 | 0.137 ± 0.027 | 0.004 ± 0.002 | 0.000 | 0.000 |
| LF-Metropolis | 0.656 ± 0.175 | 0.272 ± 0.080 | 0.077 ± 0.149 | 0.000 | 0.000 |
| Su-Vaidya | 0.651 ± 0.234 | 0.113 ± 0.020 | 0.003 ± 0.002 | 0.000 | 0.000 |
| Q-consensus | 0.292 ± 0.188 | 5.654 ± 1.016 | 0.497 ± 0.242 | 0.507 | 0.000 |
| Q-consensus (fixed beta) | 0.565 ± 0.147 | 0.108 ± 0.019 | 0.007 ± 0.006 | 0.000 | 0.000 |
| DRC-RSSG | 0.640 ± 0.370 | 0.182 ± 0.088 | 0.004 ± 0.003 | 0.000 | 0.000 |

### F = 2, attack: stealthy (fake 50)

| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |
|---|---|---|---|---|---|
| DGD | 6.632 ± 1.604 | 0.315 ± 0.107 | 0.056 ± 0.022 | 5.172 | 0.000 |
| LF | 0.692 ± 0.247 | 0.136 ± 0.028 | 0.007 ± 0.005 | 0.000 | 0.000 |
| LF-Metropolis | 0.794 ± 0.228 | 0.297 ± 0.122 | 0.032 ± 0.059 | 0.000 | 0.000 |
| Su-Vaidya | 0.685 ± 0.250 | 0.116 ± 0.026 | 0.004 ± 0.002 | 0.000 | 0.000 |
| Q-consensus | 0.185 ± 0.128 | 5.458 ± 3.477 | 0.294 ± 0.148 | 8.482 | 0.000 |
| Q-consensus (fixed beta) | 2.274 ± 1.459 | 0.150 ± 0.055 | 0.674 ± 0.480 | 0.593 | 0.000 |
| DRC-RSSG | 0.200 ± 0.111 | 0.156 ± 0.076 | 0.001 ± 0.001 | 0.000 | 0.000 |

### F = 2, attack: trim-hugger

| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |
|---|---|---|---|---|---|
| DGD | 0.398 ± 0.081 | 0.157 ± 0.047 | 0.003 ± 0.001 | 0.000 | 0.000 |
| LF | 0.666 ± 0.275 | 0.150 ± 0.039 | 0.004 ± 0.002 | 0.000 | 0.000 |
| LF-Metropolis | 0.682 ± 0.224 | 0.294 ± 0.118 | 0.019 ± 0.021 | 0.000 | 0.000 |
| Su-Vaidya | 0.618 ± 0.280 | 0.117 ± 0.027 | 0.003 ± 0.002 | 0.000 | 0.000 |
| Q-consensus | 0.265 ± 0.187 | 5.751 ± 1.633 | 0.525 ± 0.214 | 0.000 | 0.000 |
| Q-consensus (fixed beta) | 0.365 ± 0.146 | 0.114 ± 0.026 | 0.007 ± 0.004 | 0.000 | 0.000 |
| DRC-RSSG | 0.701 ± 0.350 | 0.216 ± 0.152 | 0.006 ± 0.003 | 0.000 | 0.000 |

### F = 2, attack: switching

| Method | Error | Final spread | Oscillation (TV of mean, t ≥ T/2) | Max hull violation (t ≥ T/2) | Max excess over initial range (α = 0) |
|---|---|---|---|---|---|
| DGD | 0.266 ± 0.141 | 0.157 ± 0.042 | 1.857 ± 0.141 | 0.000 | 0.000 |
| LF | 0.390 ± 0.351 | 0.157 ± 0.035 | 2.811 ± 0.474 | 0.000 | 0.000 |
| LF-Metropolis | 0.459 ± 0.289 | 0.303 ± 0.116 | 2.773 ± 0.507 | 0.000 | 0.000 |
| Su-Vaidya | 0.412 ± 0.270 | 0.120 ± 0.025 | 2.548 ± 0.552 | 0.000 | 0.000 |
| Q-consensus | 0.929 ± 0.543 | 0.656 ± 0.387 | 8.854 ± 5.073 | 0.000 | 0.000 |
| Q-consensus (fixed beta) | 0.181 ± 0.095 | 0.113 ± 0.027 | 1.981 ± 0.491 | 0.000 | 0.000 |
| DRC-RSSG | 0.432 ± 0.313 | 0.205 ± 0.106 | 2.787 ± 0.486 | 0.000 | 0.000 |

## Ex. 7.2 graph (K5, switching attack of [1])

| Method | Final error | Oscillation (TV, t ≥ 1500) | Attacker switches | Hull violation (t ≥ 10) |
|---|---|---|---|---|
| DGD | 0.446 | 0.002 | 0 | 0.000 |
| LF | 0.375 | 13.929 | 25 | 0.000 |
| LF-Metropolis | 1.443 | 16.993 | 28 | 0.000 |
| Su-Vaidya | 1.013 | 16.802 | 34 | 0.000 |
| Q-consensus | 1.997 | 3.955 | 2 | 0.000 |
| Q-consensus (fixed beta) | 0.529 | 0.045 | 0 | 0.000 |
| DRC-RSSG | 0.387 | 12.410 | 24 | 0.000 |

## U–W graph (Ex. 8.2, K = 5, b = 4, no adversaries)

| Method | Error | Bound \|T\|/n·b = b/4 |
|---|---|---|
| DGD | 0.000 | 1.00 |
| LF | 0.990 | 1.00 |
| LF-Metropolis | 0.947 | 1.00 |
| Su-Vaidya | 0.993 | 1.00 |
| Q-consensus | 0.910 | 1.00 |
| Q-consensus (fixed beta) | 0.696 | 1.00 |
| DRC-RSSG | 0.993 | 1.00 |
