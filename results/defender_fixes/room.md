# How much room is there? Oracle weights inside the shield

Mean error on the Phase 3 benchmark instances (10 graphs per F, T = 2000). The oracle tunes a free score for every node and neighbour slot, per instance, knowing x*; it is the best any shielded defender could do against that attack. c is the floor constant of the shield.

| F | Attack | LF | Oracle, c = 0.25 | Oracle, c = 0.05 | Room (c = 0.25) | Room (c = 0.05) |
|---|---|---|---|---|---|---|
| 1 | none | 0.137 | 0.000 | 0.000 | 100% | 100% |
| 1 | fixed +50 | 0.277 | 0.000 | 0.000 | 100% | 100% |
| 1 | trim-hugger | 0.303 | 0.000 | 0.000 | 100% | 100% |
| 1 | RL attacker (vs LF) | 0.304 | 0.000 | 0.000 | 100% | 100% |
| 2 | none | 0.291 | 0.000 | 0.000 | 100% | 100% |
| 2 | fixed +50 | 0.692 | 0.000 | 0.000 | 100% | 100% |
| 2 | trim-hugger | 0.666 | 0.000 | 0.000 | 100% | 100% |
| 2 | RL attacker (vs LF) | 0.735 | 0.000 | 0.000 | 100% | 100% |
