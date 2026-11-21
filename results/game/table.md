# Zero-sum game

Exploitability = mean error the round's best-response RL attacker causes on 100 unseen graphs per F (it picks the worse push direction per instance). D_0 is the memory + fallback defender; each later D_r is trained against the scripted mix plus all attackers so far.

| Round r | Defender | Exploitability, F = 1 | Exploitability, F = 2 | Max hull violation | Max excess (α = 0) |
|---|---|---|---|---|---|
| 0 | D_0 | 0.582 | 0.829 | 0.000 | 0.000 |
| 1 | D_1 | 0.581 | 0.814 | 0.000 | 0.000 |
| 2 | D_2 | 0.574 | 0.821 | 0.000 | 0.000 |
| 3 | D_3 | 0.575 | 0.794 | 0.000 | 0.000 |
| — | plain LF (its RL attacker) | 0.609 | 0.889 | — | — |
