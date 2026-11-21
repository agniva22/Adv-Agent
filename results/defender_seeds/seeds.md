# Seeds and ablations of memory + fallback

100 unseen graphs per F (the Phase 7 ones), T = 2000. Mean is over the scripted attacks and the method's own best-response RL attacker; 'own BR' is that attacker alone. p: paired Wilcoxon test over the graphs (two-sided); 'better' counts graphs where the method has the lower error.

## F = 1

| Method | No attack | Mean over attacks | Own BR | Safe? | Own BR vs LF: better, p | Own BR vs Trimmed mean: better, p | Mean vs LF: better, p |
|---|---|---|---|---|---|---|---|
| LF | 0.215 | 0.403 | 0.609 | yes | — | 38/100, 5.1e-02 | — |
| Su-Vaidya | 0.220 | 0.447 | 0.672 | yes | 37/100, 8.7e-03 | 39/100, 6.6e-03 | 45/100, 1.0e-01 |
| Trimmed mean | 0.231 | 0.399 | 0.602 | yes | 62/100, 5.1e-02 | — | 51/100, 3.3e-01 |
| memory + fallback | 0.182 | 0.378 | 0.582 | yes | 57/100, 9.1e-02 | 54/100, 3.0e-01 | 62/100, 3.9e-03 |
| memory + fallback (seed 1) | 0.169 | 0.379 | 0.582 | yes | 57/100, 1.7e-01 | 55/100, 3.9e-01 | 60/100, 2.7e-02 |
| memory + fallback (seed 2) | 0.173 | 0.393 | 0.575 | yes | 56/100, 4.6e-02 | 52/100, 2.0e-01 | 53/100, 2.6e-01 |
| memory only | 0.182 | 0.381 | 0.576 | yes | 56/100, 3.7e-02 | 56/100, 2.3e-01 | 58/100, 3.1e-02 |
| fallback only | 0.188 | 0.384 | 0.588 | yes | 57/100, 1.1e-01 | 57/100, 3.7e-01 | 56/100, 2.1e-02 |
| memory + fallback, 3 seeds averaged | 0.175 | 0.383 | 0.580 | yes | 55/100, 7.0e-02 | 51/100, 2.8e-01 | 60/100, 2.9e-02 |

Over the 3 seeds of memory + fallback (mean ± std of the seed means): no attack 0.175 ± 0.005, mean over attacks 0.383 ± 0.007, own BR 0.580 ± 0.004.

## F = 2

| Method | No attack | Mean over attacks | Own BR | Safe? | Own BR vs LF: better, p | Own BR vs Trimmed mean: better, p | Mean vs LF: better, p |
|---|---|---|---|---|---|---|---|
| LF | 0.277 | 0.599 | 0.889 | yes | — | 36/100, 1.1e-03 | — |
| Su-Vaidya | 0.249 | 0.600 | 0.956 | yes | 37/100, 1.3e-02 | 33/100, 3.6e-03 | 47/100, 8.4e-01 |
| Trimmed mean | 0.289 | 0.587 | 0.871 | yes | 64/100, 1.1e-03 | — | 52/100, 1.4e-01 |
| memory + fallback | 0.216 | 0.565 | 0.829 | yes | 69/100, 9.8e-06 | 60/100, 3.7e-03 | 61/100, 5.1e-03 |
| memory + fallback (seed 1) | 0.189 | 0.545 | 0.790 | yes | 77/100, 2.6e-07 | 67/100, 3.8e-05 | 63/100, 7.6e-04 |
| memory + fallback (seed 2) | 0.200 | 0.568 | 0.812 | yes | 71/100, 2.6e-06 | 61/100, 7.5e-04 | 59/100, 2.5e-02 |
| memory only | 0.214 | 0.575 | 0.825 | yes | 72/100, 2.6e-06 | 60/100, 1.6e-03 | 60/100, 2.9e-02 |
| fallback only | 0.228 | 0.574 | 0.844 | yes | 66/100, 1.1e-04 | 57/100, 2.3e-02 | 59/100, 4.9e-02 |
| memory + fallback, 3 seeds averaged | 0.202 | 0.559 | 0.810 | yes | 73/100, 8.3e-07 | 63/100, 2.6e-04 | 62/100, 3.7e-03 |

Over the 3 seeds of memory + fallback (mean ± std of the seed means): no attack 0.202 ± 0.011, mean over attacks 0.559 ± 0.010, own BR 0.810 ± 0.016.

