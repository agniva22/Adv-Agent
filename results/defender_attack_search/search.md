# Stronger attacker search against the memory + fallback defender

## unseen (100 graphs per F)

Mean error per learned attacker (worst push direction per graph; swing also with sampled actions). 'Worst' is the worst attacker for that method, by mean error.

### F = 1

| Attacker | LF | Su-Vaidya | memory + fallback |
|---|---|---|---|
| Phase 4 (vs LF) | 0.609 | 0.601 | 0.583 |
| best response, warm start (vs LF) | 0.609 | 0.602 | 0.582 |
| best response, warm start (vs memory + fallback) | 0.608 | 0.601 | 0.582 |
| from scratch (vs LF) | 0.609 | 0.601 | 0.583 |
| from scratch (vs memory + fallback) | 0.609 | 0.602 | 0.583 |
| swing (vs LF) | 0.448 | 0.459 | 0.440 |
| swing (vs memory + fallback) | 0.379 | 0.369 | 0.352 |
| memory-aware (vs memory + fallback) | 0.609 | 0.601 | 0.584 |
| **worst** | **0.609** (from scratch (vs memory + fallback)) | **0.602** (from scratch (vs memory + fallback)) | **0.584** (memory-aware (vs memory + fallback)) |
| swing: oscillation of x̄ (t ≥ T/2), worst | 5.053 | 4.784 | 4.745 |

Safe? LF: yes; Su-Vaidya: yes; memory + fallback: yes

### F = 2

| Attacker | LF | Su-Vaidya | memory + fallback |
|---|---|---|---|
| Phase 4 (vs LF) | 0.889 | 0.864 | 0.826 |
| best response, warm start (vs LF) | 0.889 | 0.865 | 0.828 |
| best response, warm start (vs memory + fallback) | 0.889 | 0.865 | 0.828 |
| from scratch (vs LF) | 0.883 | 0.860 | 0.819 |
| from scratch (vs memory + fallback) | 0.888 | 0.864 | 0.824 |
| swing (vs LF) | 0.325 | 0.342 | 0.266 |
| swing (vs memory + fallback) | 0.439 | 0.445 | 0.343 |
| memory-aware (vs memory + fallback) | 0.888 | 0.866 | 0.833 |
| **worst** | **0.889** (best response, warm start (vs LF)) | **0.866** (memory-aware (vs memory + fallback)) | **0.833** (memory-aware (vs memory + fallback)) |
| swing: oscillation of x̄ (t ≥ T/2), worst | 1.119 | 1.029 | 1.025 |

Safe? LF: yes; Su-Vaidya: yes; memory + fallback: yes

## benchmark (10 graphs per F)

Mean error per learned attacker (worst push direction per graph; swing also with sampled actions). 'Worst' is the worst attacker for that method, by mean error.

### F = 1

| Attacker | LF | Su-Vaidya | memory + fallback |
|---|---|---|---|
| Phase 4 (vs LF) | 0.539 | 0.516 | 0.522 |
| best response, warm start (vs LF) | 0.539 | 0.515 | 0.521 |
| best response, warm start (vs memory + fallback) | 0.538 | 0.514 | 0.520 |
| from scratch (vs LF) | 0.539 | 0.515 | 0.522 |
| from scratch (vs memory + fallback) | 0.539 | 0.515 | 0.522 |
| swing (vs LF) | 0.313 | 0.316 | 0.308 |
| swing (vs memory + fallback) | 0.395 | 0.378 | 0.392 |
| memory-aware (vs memory + fallback) | 0.539 | 0.515 | 0.522 |
| **worst** | **0.539** (memory-aware (vs memory + fallback)) | **0.516** (Phase 4 (vs LF)) | **0.522** (from scratch (vs memory + fallback)) |
| swing: oscillation of x̄ (t ≥ T/2), worst | 4.468 | 4.311 | 4.060 |

Safe? LF: yes; Su-Vaidya: yes; memory + fallback: yes

### F = 2

| Attacker | LF | Su-Vaidya | memory + fallback |
|---|---|---|---|
| Phase 4 (vs LF) | 0.898 | 0.849 | 0.858 |
| best response, warm start (vs LF) | 0.900 | 0.849 | 0.859 |
| best response, warm start (vs memory + fallback) | 0.899 | 0.849 | 0.859 |
| from scratch (vs LF) | 0.902 | 0.845 | 0.853 |
| from scratch (vs memory + fallback) | 0.900 | 0.849 | 0.855 |
| swing (vs LF) | 0.332 | 0.320 | 0.324 |
| swing (vs memory + fallback) | 0.428 | 0.414 | 0.384 |
| memory-aware (vs memory + fallback) | 0.894 | 0.848 | 0.858 |
| **worst** | **0.902** (from scratch (vs LF)) | **0.849** (best response, warm start (vs LF)) | **0.859** (best response, warm start (vs memory + fallback)) |
| swing: oscillation of x̄ (t ≥ T/2), worst | 1.190 | 1.099 | 1.035 |

Safe? LF: yes; Su-Vaidya: yes; memory + fallback: yes

