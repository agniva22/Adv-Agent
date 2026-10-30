# Shield theorem (Phase 5, contribution C3)

**Claim.** A learned defender that only re-weights the values LF keeps, with a floored softmax, inherits every guarantee of Sundaram & Gharesifard [1]. This holds for *every* parameter setting, so it also holds during training and against an attacker that exploits the learned policy.

## Setting

[1, Sec. V] defines the LF dynamics (8). At each step t, every regular node v_i:

1. receives the values {x_j(t) : v_j ∈ N_i⁻};
2. removes the F highest values above its own and the F lowest values below its own (all of them if there are fewer than F), breaking ties arbitrarily. The retained in-neighbours form J_i(t);
3. updates

   x_i(t+1) = a_ii(t) x_i(t) + Σ_{j ∈ J_i(t)} a_ij(t) x_j(t) − α_t d_i(t),

   where d_i(t) is a subgradient of f_i at the convex combination a_ii x_i + Σ a_ij x_j.

The only requirement on the weights is: *"at each time-step t and for each v_i ∈ R, the weights a_ij(t), v_j ∈ {v_i} ∪ J_i(t), are lower-bounded by some strictly positive real number η and sum to 1"* [1, Sec. V]. The weights may change over time and are not tied to any formula. Nothing in [1] restricts how they are chosen.

## The shielded rule (`advagent/learn.py`, `ShieldedDefender`)

Node v_i computes J_i(t) with the LF filter (step 2 above, `torchsim.lf_keep`). A policy π_θ then outputs finite scores s_ij(t) for j ∈ {i} ∪ J_i(t). The weights are

  a_ij(t) = η_i + (1 − m_i(t) η_i) · softmax_j(s_i(t)),  with  m_i(t) = |J_i(t)| + 1,  η_i = c / (|N_i⁻| + 1),  0 < c ≤ 1,

and a_ij(t) = 0 for every in-neighbour that LF removed. The node then takes the subgradient step at the resulting convex combination.

The scores may depend on anything: the node's own state, the received values, its gradient, the step size, its history, or the adversaries' values. Our implementation uses its own state, the retained values, degrees, its gradient and α_t.

## Proposition

Let d_max be the largest in-degree and η = c / (d_max + 1). For every θ, and more generally for any rule that produces finite scores, the regular nodes running the shielded rule follow the LF dynamics (8) of [1] with parameter F and weight bound η. Hence:

- **Thm. 6.1:** with an F-local set of Byzantine nodes on a (2F+1)-robust graph, subgradients bounded by L and α_t → 0, the regular nodes reach consensus.
- **Thm. 6.4:** with F-total malicious nodes, they reach consensus on every (F+1, F+1)-robust graph.
- **Thm. 7.1:** under either of these conditions, with Σ α_t = ∞ and α_t → 0, every regular state ends in the hull [min_i min M_i, max_i max M_i] of the regular nodes' local minimizers, whatever the adversaries do.

All three hold for any learned policy, including one that is still training or one that an adversary has learned to exploit.

## Proof

Check the four requirements of (8) at an arbitrary step t and regular node v_i.

1. **Support.** The weights are nonzero only on {i} ∪ J_i(t), and J_i(t) is exactly LF's retained set.
2. **Lower bound.** softmax_j ≥ 0 and 1 − m_i η_i ≥ 1 − (|N_i⁻| + 1) η_i = 1 − c ≥ 0, so a_ij(t) ≥ η_i ≥ c / (d_max + 1) = η > 0. Here m_i ≤ |N_i⁻| + 1 because J_i(t) ⊆ N_i⁻.
3. **Sum to one.** Σ_j a_ij = m_i η_i + (1 − m_i η_i) Σ_j softmax_j = 1.
4. **Subgradient point.** The simulator evaluates d_i at the convex combination (`torchsim.step`, gradient at y unless the rule asks for x_i).

So the closed loop is an instance of (8) with parameter F and bound η.

The hypotheses of Thms. 6.1, 6.4 and 7.1 concern only the graph, the adversary model, the f_i, the step sizes and the fact that the regular nodes run (8) with weights bounded by η. Their proofs use only per-step properties of the weights:

- the reduction to regular-only dynamics (9) in Prop. 5.1 uses a_ij ≥ η and Σ a_ij = 1;
- the consensus arguments use lower bounds of η or η/2 on the entries of the equivalent matrix Ā(t).

None of them uses how the weights were chosen. The conclusions therefore apply. ∎

## Remarks

- **What is not inherited.** [1] gives no optimality guarantee for LF, and none is claimed here. The lower bound of [1, Prop. 8.1] applies to the shielded rule too, as to any rule that respects the hull. The learned weights can only move the result *within* the guaranteed set.
- **The trade-off set by c.** c = 1 forces uniform weights, which is plain LF. A smaller c gives the policy more freedom but a smaller η, which weakens the consensus rate in the proofs. We use c = 0.25, so every weight is at least a quarter of LF's uniform weight 1/(|N_i⁻| + 1).
- **Implementation guard.** The proof needs finite scores. Non-finite scores would make the softmax undefined, so the implementation replaces NaN and ±∞ scores before the softmax (`torch.nan_to_num`).
- **Ablation (unshielded).** `ShieldedDefender(shield=False)` drops step 2 and the floor. It is not an instance of (8), so none of the above applies. Phase 7 measures what that costs.
- **Empirical check.** `tests/test_phase5.py` runs randomly initialised policies with large parameters (far from LF) against the strongest scripted attacks and checks hull containment and consensus. The Phase 7 evaluation reports zero violations for the trained policies.
