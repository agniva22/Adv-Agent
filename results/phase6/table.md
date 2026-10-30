# Phase 6: zero-sum training

Exploitability = mean error the round's best-response RL attacker causes on the Phase 3 benchmark instances (it picks the worse push direction per instance). D_0 is the Phase 5 shielded defender; each later D_r is trained against the scripted mix plus all attackers so far.

| Round r | Defender | Exploitability, F = 1 | Exploitability, F = 2 | Max hull violation | Max excess (α = 0) |
|---|---|---|---|---|---|
| 0 | D_0 | 0.539 | 0.898 | 0.000 | 0.000 |
| 1 | D_1 | 0.535 | 0.903 | 0.000 | 0.000 |
| 2 | D_2 | 0.539 | 0.904 | 0.000 | 0.000 |
| 3 | D_3 | 0.537 | 0.899 | 0.000 | 0.000 |
| — | plain LF (Phase 4 attacker) | 0.539 | 0.898 | — | — |
