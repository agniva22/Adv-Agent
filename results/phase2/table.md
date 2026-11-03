# Phase 2 baseline comparison

10 seeds; random 20-node 3-robust graphs; F = 1; T = 2000; alpha_t = 0.5/(t+1)^0.6. Mean ± std over seeds; violations are worst case.

### Attack: none

| Method | Error \|x̄ − x*\| | Final spread D(T) | Max hull violation (t ≥ T/2) | Safety: excess over initial range (α = 0) |
|---|---|---|---|---|
| DGD | 0.000 ± 0.000 | 0.185 ± 0.048 | 0.000 | 0.000 |
| LF | 0.171 ± 0.137 | 0.155 ± 0.027 | 0.000 | 0.000 |
| LF-Metropolis | 0.166 ± 0.118 | 0.282 ± 0.079 | 0.000 | 0.000 |
| Su-Vaidya | 0.141 ± 0.131 | 0.066 ± 0.013 | 0.000 | 0.000 |
| Trimmed mean | 0.161 ± 0.140 | 0.135 ± 0.022 | 0.000 | 0.000 |
| Q-consensus | 0.193 ± 0.095 | 5.152 ± 1.938 | 0.000 | 0.000 |
| Q-consensus (fixed beta) | 0.080 ± 0.052 | 0.132 ± 0.020 | 0.000 | 0.000 |
| DRC-RSSG | 0.226 ± 0.129 | 0.148 ± 0.028 | 0.000 | 0.000 |

### Attack: fixed +50

| Method | Error \|x̄ − x*\| | Final spread D(T) | Max hull violation (t ≥ T/2) | Safety: excess over initial range (α = 0) |
|---|---|---|---|---|
| DGD | 40.607 ± 1.209 | 2.818 ± 0.907 | 39.753 | 47.259 |
| LF | 0.538 ± 0.164 | 0.145 ± 0.031 | 0.000 | 0.000 |
| LF-Metropolis | 0.527 ± 0.161 | 0.284 ± 0.107 | 0.000 | 0.000 |
| Su-Vaidya | 0.431 ± 0.166 | 0.070 ± 0.020 | 0.000 | 0.000 |
| Trimmed mean | 0.545 ± 0.159 | 0.131 ± 0.029 | 0.000 | 0.000 |
| Q-consensus | 2.186 ± 0.699 | 16.512 ± 2.862 | 13.405 | 47.259 |
| Q-consensus (fixed beta) | 0.072 ± 0.047 | 0.126 ± 0.023 | 4.429 | 47.259 |
| DRC-RSSG | 0.140 ± 0.087 | 0.127 ± 0.026 | 0.000 | 0.000 |

### Attack: random Byzantine

| Method | Error \|x̄ − x*\| | Final spread D(T) | Max hull violation (t ≥ T/2) | Safety: excess over initial range (α = 0) |
|---|---|---|---|---|
| DGD | 1.241 ± 1.052 | 9.978 ± 2.316 | 10.050 | 10.685 |
| LF | 0.123 ± 0.083 | 0.143 ± 0.032 | 0.000 | 0.000 |
| LF-Metropolis | 0.096 ± 0.114 | 0.266 ± 0.077 | 0.000 | 0.000 |
| Su-Vaidya | 0.139 ± 0.060 | 0.069 ± 0.021 | 0.000 | 0.000 |
| Trimmed mean | 0.127 ± 0.076 | 0.126 ± 0.028 | 0.000 | 0.000 |
| Q-consensus | 0.231 ± 0.158 | 3.603 ± 2.313 | 0.000 | 10.030 |
| Q-consensus (fixed beta) | 0.123 ± 0.093 | 0.330 ± 0.077 | 0.000 | 10.022 |
| DRC-RSSG | 0.132 ± 0.094 | 0.127 ± 0.026 | 0.000 | 0.000 |

### Attack: farthest evader

| Method | Error \|x̄ − x*\| | Final spread D(T) | Max hull violation (t ≥ T/2) | Safety: excess over initial range (α = 0) |
|---|---|---|---|---|
| DGD | 0.521 ± 0.197 | 0.221 ± 0.092 | 0.000 | 0.000 |
| LF | 0.511 ± 0.156 | 0.146 ± 0.031 | 0.000 | 0.000 |
| LF-Metropolis | 0.472 ± 0.175 | 0.289 ± 0.103 | 0.000 | 0.000 |
| Su-Vaidya | 0.422 ± 0.166 | 0.070 ± 0.020 | 0.000 | 0.000 |
| Trimmed mean | 0.521 ± 0.155 | 0.132 ± 0.029 | 0.000 | 0.000 |
| Q-consensus | 0.183 ± 0.106 | 4.603 ± 2.143 | 0.000 | 0.000 |
| Q-consensus (fixed beta) | 0.552 ± 0.159 | 0.146 ± 0.034 | 0.000 | 0.000 |
| DRC-RSSG | 0.935 ± 0.307 | 0.188 ± 0.065 | 0.000 | 1.777 |
