# Shield Theorem

> **Claim.** A learned defender that only re-weights the values LF keeps, using a floored softmax, inherits every guarantee of Sundaram and Gharesifard [1]. This holds for *every* parameter setting of the policy, so it also holds during training and against an attacker that has learned to exploit the policy.

This is the theoretical core of the project (contribution C3 in the [proposal](../README.md)). The code is `ShieldedDefender` in `advagent/learn.py`, and the results are in [IMPLEMENTATION.md](../IMPLEMENTATION.md), Phases 5–7.

## 1. Setting

We follow [1, Sec. V]. Let $\mathcal{R}$ be the set of regular nodes, and let $\mathcal{N}_i^-$ be the in-neighbours of node $v_i$, that is, the nodes whose values $v_i$ receives. At every step $t$, each regular node $v_i$ does three things:

1. it receives the values $\{x_j(t) : v_j \in \mathcal{N}_i^-\}$;
2. it removes the $F$ highest values above its own and the $F$ lowest values below its own (all of them if there are fewer than $F$), breaking ties arbitrarily, and calls the retained in-neighbours $\mathcal{J}_i(t) \subseteq \mathcal{N}_i^-$;
3. it updates

$$
x_i(t+1) = a_{ii}(t)\,x_i(t) + \sum_{j \in \mathcal{J}_i(t)} a_{ij}(t)\,x_j(t) - \alpha_t\, d_i(t), \tag{1}
$$

where $d_i(t)$ is a subgradient of $f_i$ at the convex combination $a_{ii}(t)\,x_i(t) + \sum_{j \in \mathcal{J}_i(t)} a_{ij}(t)\,x_j(t)$.

Equation (1) is eq. (8) of [1]. The paper places only one requirement on the weights: *"at each time-step $t$ and for each $v_i \in \mathcal{R}$, the weights $a_{ij}(t)$, $v_j \in \{v_i\} \cup \mathcal{J}_i(t)$, are lower-bounded by some strictly positive real number $\eta$ and sum to 1"* [1, Sec. V]. The weights may change over time and do not have to follow any formula. Nothing in [1] restricts how they are chosen, and that freedom is what we use.

## 2. The shielded rule

Fix a constant $0 < c \le 1$. At each step $t$, node $v_i$ first computes $\mathcal{J}_i(t)$ with the LF filter (step 2 above). A policy $\pi_\theta$ then gives a finite score $s_{ij}(t)$ to the node itself and to every retained neighbour, $j \in \{i\} \cup \mathcal{J}_i(t)$. Let

$$
m_i(t) = \lvert\mathcal{J}_i(t)\rvert + 1, \qquad \eta_i = \frac{c}{\lvert\mathcal{N}_i^-\rvert + 1}, \qquad p_{ij}(t) = \frac{e^{s_{ij}(t)}}{\sum_{k \in \{i\} \cup \mathcal{J}_i(t)} e^{s_{ik}(t)}} .
$$

The weights are a *floored softmax*:

$$
a_{ij}(t) =
\begin{cases}
\eta_i + \bigl(1 - m_i(t)\,\eta_i\bigr)\, p_{ij}(t), & j \in \{i\} \cup \mathcal{J}_i(t), \\
0, & j \in \mathcal{N}_i^- \setminus \mathcal{J}_i(t).
\end{cases} \tag{2}
$$

In words, every value LF keeps first gets a guaranteed share $\eta_i$, and the policy only decides how to split the rest. The node then takes the subgradient step of (1) at the resulting convex combination.

The scores may depend on anything: the node's own state, the received values, its gradient, the step size, its history, or even the adversaries' values. Our implementation uses the node's own state, the retained values, the degrees, the gradient and $\alpha_t$.

## 3. Proposition

> **Proposition (Shield).** Let $d_{\max} = \max_{i \in \mathcal{R}} \lvert\mathcal{N}_i^-\rvert$ and $\eta = c/(d_{\max}+1)$. For every $\theta$, and more generally for any rule that produces finite scores, the regular nodes running the shielded rule (2) follow the LF dynamics (1) with parameter $F$ and weight bound $\eta$. Consequently:
>
> - **[1, Thm. 6.1].** If the adversaries form an $F$-local set of Byzantine nodes, the graph is $(2F+1)$-robust, the subgradients are bounded by $L$ and $\alpha_t \to 0$, then the regular nodes reach consensus.
> - **[1, Thm. 6.4].** If the adversaries form an $F$-total set of malicious nodes and the graph is $(F+1, F+1)$-robust, then the regular nodes reach consensus.
> - **[1, Thm. 7.1].** Under either condition, if also $\sum_t \alpha_t = \infty$ and $\alpha_t \to 0$, every regular state converges into the interval $\bigl[\min_{i \in \mathcal{R}} \min \mathcal{M}_i,\ \max_{i \in \mathcal{R}} \max \mathcal{M}_i\bigr]$, where $\mathcal{M}_i$ is the set of minimizers of $f_i$, whatever the adversaries do.
>
> All three hold for any learned policy, including one that is still training and one that an adversary has learned to exploit.

## 4. Proof

Fix any step $t$ and any regular node $v_i$. Because the scores are finite, the softmax terms are well defined, with $p_{ij}(t) > 0$ and $\sum_{j} p_{ij}(t) = 1$. We check the four requirements of (1) one by one.

**Support.** By (2), $a_{ij}(t)$ is nonzero only for $j \in \{i\} \cup \mathcal{J}_i(t)$, and $\mathcal{J}_i(t)$ is exactly the set LF keeps.

**Lower bound.** Since $\mathcal{J}_i(t) \subseteq \mathcal{N}_i^-$, we have $m_i(t) \le \lvert\mathcal{N}_i^-\rvert + 1$, and so

$$
1 - m_i(t)\,\eta_i \;\ge\; 1 - \bigl(\lvert\mathcal{N}_i^-\rvert + 1\bigr)\,\eta_i \;=\; 1 - c \;\ge\; 0 .
$$

Together with $p_{ij}(t) \ge 0$, this gives, for every $j \in \{i\} \cup \mathcal{J}_i(t)$,

$$
a_{ij}(t) \;\ge\; \eta_i \;=\; \frac{c}{\lvert\mathcal{N}_i^-\rvert + 1} \;\ge\; \frac{c}{d_{\max} + 1} \;=\; \eta \;>\; 0 .
$$

**Sum to one.**

$$
\sum_{j \in \{i\} \cup \mathcal{J}_i(t)} a_{ij}(t) \;=\; m_i(t)\,\eta_i + \bigl(1 - m_i(t)\,\eta_i\bigr) \sum_{j \in \{i\} \cup \mathcal{J}_i(t)} p_{ij}(t) \;=\; m_i(t)\,\eta_i + 1 - m_i(t)\,\eta_i \;=\; 1 .
$$

**Subgradient point.** The subgradient $d_i(t)$ is evaluated at the convex combination built from these weights, exactly as (1) requires (`torchsim.step` computes the gradient there).

So the closed loop is an instance of (1) with parameter $F$ and bound $\eta$.

It remains to check that the theorems of [1] apply to it. Their hypotheses concern only the graph, the adversary model, the functions $f_i$, the step sizes, and the fact that the regular nodes run (1) with weights bounded below by $\eta$. Their proofs also use only per-step properties of the weights. The reduction to the regular-only dynamics (9) in [1, Prop. 5.1] needs $a_{ij}(t) \ge \eta$ and $\sum_j a_{ij}(t) = 1$, and the consensus arguments need lower bounds of $\eta$ or $\eta/2$ on the entries of the equivalent matrix $\bar{A}(t)$. None of them depends on *how* the weights were chosen, so all three conclusions hold. $\blacksquare$

## 5. Remarks

**What is not inherited.** [1] gives no optimality guarantee for LF, and we claim none here. The lower bound of [1, Prop. 8.1] applies to the shielded rule just as it does to any rule that keeps the regular states in the hull: on some instances the error is at least $\frac{\lvert T\rvert}{n}\lvert b - a\rvert$, where $T$ is a maximum $F$-local set. The learned weights can only move the result *within* the guaranteed set.

**Relation to plain LF.** If all of node $v_i$'s scores are equal, then $p_{ij}(t) = 1/m_i(t)$ and

$$
a_{ij}(t) = \eta_i + \frac{1 - m_i(t)\,\eta_i}{m_i(t)} = \frac{1}{m_i(t)},
$$

which is exactly LF's uniform weight over $\{i\} \cup \mathcal{J}_i(t)$, for any $c$. Our policy's last layer starts at zero, so training starts from plain LF.

**Memory and the LF fallback.** The memory + fallback defender (`TrustDefender` in `advagent/learn.py`) adds two things to rule (2), and neither leaves the proposition.

1. *Memory.* Each node keeps a running average of how far each neighbour's value was from its own, how often LF trimmed it, and on which side it was. These averages are extra inputs to the scores. The proposition allows scores that depend on anything, including the history, and its proof uses only properties of the weights at a single step. Scores that depend on memory are therefore covered. The same holds for an attacker that watches the memory (`MemoryAwareAttacker`): the guarantees of [1] hold against any adversary behaviour.
2. *Fallback.* A gate $g_i(t) \in [0, 1]$, computed from the same inputs, blends the shielded weights with LF's uniform weights:
$$
\tilde a_{ij}(t) = g_i(t)\, a_{ij}(t) + \bigl(1 - g_i(t)\bigr)\, \frac{1}{m_i(t)}, \qquad j \in \{i\} \cup \mathcal{J}_i(t),
$$
and $\tilde a_{ij}(t) = 0$ otherwise. Both weight vectors have the same support and sum to one. Both are also at least $\eta_i$: for the first this is the proof above, and for the second $1/m_i(t) \ge 1/(\lvert\mathcal{N}_i^-\rvert + 1) \ge \eta_i$, since $c \le 1$. Any convex combination keeps these three properties. So $\tilde a_{ij}(t)$ is again an instance of (1) with bound $\eta$, whatever the gate outputs. As for the scores, the implementation makes the gate's input to the sigmoid finite (`torch.nan_to_num`), so $g_i(t)$ is always defined.

**The trade-off set by $c$.** The policy controls only the mass $1 - m_i(t)\,\eta_i$ above the floor. A larger $c$ raises the floor $\eta$, which strengthens the consensus rate in the proofs but leaves the policy less freedom. When LF removes nothing ($m_i = \lvert\mathcal{N}_i^-\rvert + 1$), $c = 1$ leaves no freedom at all. A smaller $c$ gives the policy more freedom at the price of a smaller $\eta$. The Phase 5 shielded defender uses $c = 0.25$, so every weight is at least a quarter of $1/(\lvert\mathcal{N}_i^-\rvert + 1)$. The memory + fallback defender uses $c = 0.05$. This gives the policy more freedom; the fallback gate can always move it back towards LF.

**Implementation guard.** The proof needs finite scores, since a non-finite score would make the softmax undefined. The implementation therefore replaces NaN and $\pm\infty$ scores before the softmax (`torch.nan_to_num`).

**Unshielded ablation.** `ShieldedDefender(shield=False)` drops both the filter and the floor. It is not an instance of (1), so the proposition does not apply to it. In our experiments a constant $+50$ attack captures it (mean error 39 for $F = 1$), while the shielded defender is never unsafe.

**Empirical check.** `tests/test_phase5.py` runs randomly initialised policies with large parameters, far from LF, against the strongest scripted attacks, and checks hull containment and consensus. The Phase 5–7 evaluations report zero safety violations for every trained shielded policy.

## Reference

**[1]** S. Sundaram and B. Gharesifard, "Distributed optimization under adversarial nodes," *IEEE Trans. Autom. Control*, vol. 64, no. 3, pp. 1063–1076, Mar. 2019. https://doi.org/10.1109/TAC.2018.2836919
