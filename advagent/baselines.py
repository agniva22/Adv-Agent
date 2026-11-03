"""Comparison methods (Phase 2). Each is an aggregation rule for ``dynamics.simulate``.

Q-consensus [7] and DRC-RSSG [10] were proposed for pure consensus; here they
form the consensus point y_i and the simulator adds the same subgradient step
as every other method. That adaptation is ours and is stated in the paper.
"""

import numpy as np
from scipy.optimize import linprog

from .dynamics import lf_keep


class LFMetropolis:
    """LF filtering with Metropolis weights a_ij = 1/(1 + max(d_i, d_j)) on the
    retained set, the self-weight taking the rest.

    Every weight is at least 1/(1 + max degree) and they sum to one, so this
    rule stays inside the set where the theorems of [1] hold. It isolates how
    much of LF's bias comes from its uniform weights (IMPLEMENTATION.md, finding 2).
    """

    name = "LF-Metropolis"

    def __init__(self, F):
        self.F = F

    def aggregate(self, i, own, vals, ctx):
        adj, nb = ctx["adj"], ctx["nbrs"][i]
        deg = adj.sum(axis=1)
        keep = lf_keep(own, vals, self.F)
        w = 1.0 / (1 + np.maximum(deg[i], deg[nb[keep]]))
        return own + (w * (vals[keep] - own)).sum()


class TrimmedMean:
    """A simple trimmed-mean update (until 2026-11-03 this stood in for Su-Vaidya [5]).

    Pool the node's own value with the received ones, drop the F largest and F
    smallest of the pool, average the rest, and take the gradient step at the
    node's current state x_i(t) rather than at the average.
    """

    name = "Trimmed mean"
    grad_at_own = True

    def __init__(self, F):
        self.F = F

    def aggregate(self, i, own, vals, ctx):
        pool = np.sort(np.append(vals, own))
        if pool.size <= 2 * self.F:
            return own
        return pool[self.F: pool.size - self.F].mean()


class SuVaidya:
    """Byzantine-resilient gradient descent of Su & Vaidya [5, Alg. 1].

    Nodes exchange states and gradients (each evaluated at the sender's own
    state). Node i drops the b = F largest and F smallest received states and
    averages the rest together with its own state (always kept); it pools the
    received gradients with its own, drops the F largest and F smallest, and
    takes the midpoint of what is left; then x_i <- P_X[x_tilde - alpha_t g_tilde]
    with X = [-50, 50]. ``dynamics.simulate`` calls ``update`` instead of the
    usual aggregate-then-descend step because ``exchanges_gradients`` is set.
    """

    name = "Su-Vaidya"
    exchanges_gradients = True

    def __init__(self, F, X=(-50.0, 50.0)):
        self.F, self.X = F, X

    def update(self, i, own, vals, g_own, gvals, at, ctx):
        b = self.F
        if vals.size < 2 * b + 1:
            return own
        kept = np.sort(vals)[b: vals.size - b]
        xt = (own + kept.sum()) / (kept.size + 1)
        gp = np.sort(np.append(gvals, g_own))[b: gvals.size + 1 - b]
        return float(np.clip(xt - at * 0.5 * (gp[0] + gp[-1]), *self.X))


class QConsensus:
    """Q-consensus of Hou et al. [7, Algorithm 2].

    Each node keeps a credibility Q_ij per in-neighbour, moved towards the
    reward r_ij = exp(-beta_k |x_j - x_i| / scale) at rate eta. Weights are
    a_ij = Q_ij / sum_j Q_ij * (1 - 1/(|N_i|+1)), and the self-weight is
    1/(|N_i|+1). [7] uses beta_k = 10 + 0.1 k and eta = 0.01 for states in
    [0, 1]; ``scale`` rescales distances for other state ranges.
    """

    name = "Q-consensus"

    def __init__(self, beta0=10.0, beta1=0.1, eta=0.01, scale=1.0):
        self.beta0, self.beta1, self.eta, self.scale = beta0, beta1, eta, scale

    def reset(self, ctx):
        self.Q = np.ones(ctx["adj"].shape)

    def aggregate(self, i, own, vals, ctx):
        nb = ctx["nbrs"][i]
        beta = self.beta0 + self.beta1 * ctx["t"]
        r = np.exp(-beta * np.abs(vals - own) / self.scale)
        self.Q[i, nb] += self.eta * (r - self.Q[i, nb])
        q = self.Q[i, nb]
        self_w = 1.0 / (nb.size + 1)
        w = q / q.sum() * (1 - self_w) if q.sum() > 0 else np.full(nb.size, (1 - self_w) / nb.size)
        return self_w * own + (w * vals).sum()


def rssg_weights(dist, F, q, mu=0.05):
    """Optimal DRC-RSSG weights [10, eq. (12)] in closed form.

    ``dist`` are |x_ij - x_i| for the in-edges, ``q`` the empirical
    distribution of the attack intensity l in {0..F}. Utilities follow [10,
    Sec. IV]: the F farthest edges B_i get -B (so weight 0); other edges give
    +1 if unattacked and -1 if attacked; the self-loop gives 0. With equal
    attacker gains, the attacker hits the l largest weights, so for a fixed mass
    on the K non-B edges, spreading it evenly is optimal. Moving mass to those
    edges then pays 1 - 2 E[min(l, K)] / K per unit, which fixes whether the
    edges or the self-loop get everything above the floor ``mu``.
    Returns (edge weights, self weight).
    """
    k = dist.size
    w = np.zeros(k)
    if k <= F:
        return w, 1.0
    nonB = np.argsort(dist, kind="stable")[: k - F]
    K = nonB.size
    l = np.arange(q.size)
    gain = 1 - 2 * (q * np.minimum(l, K)).sum() / K
    mu = min(mu, 1.0 / (K + 1))
    if gain > 0:
        w[nonB] = (1 - mu) / K
        return w, mu
    w[nonB] = mu
    return w, 1 - K * mu


def rssg_lp_value(dist, F, q, mu=0.05):
    """Optimal objective of [10, eq. (12)] solved as an LP; used only to test ``rssg_weights``.

    The attacker's best response for intensity l takes the top-l weights; the
    top-l sum is written as min over u of l*u + sum_j max(w_j - u, 0).
    """
    k = dist.size
    nonB = np.argsort(dist, kind="stable")[: k - F]
    K = nonB.size
    L = q.size
    # variables: w (K), w_self, then per l: u_l, v_l (K)
    nv = K + 1 + L * (1 + K)
    c = np.zeros(nv)
    c[:K] = -1.0
    A_ub, b_ub = [], []
    for li in range(L):
        base = K + 1 + li * (1 + K)
        c[base] = 2 * q[li] * min(li, K)    # + 2 q_l * l * u_l (top-l of K values = all K if l > K)
        c[base + 1: base + 1 + K] = 2 * q[li]
        for j in range(K):                   # w_j - u_l - v_lj <= 0
            row = np.zeros(nv)
            row[j], row[base], row[base + 1 + j] = 1, -1, -1
            A_ub.append(row)
            b_ub.append(0)
    A_eq = np.zeros((1, nv))
    A_eq[0, : K + 1] = 1
    mu = min(mu, 1.0 / (K + 1))
    bounds = [(mu, 1)] * (K + 1) + [(None, None), *[(0, None)] * K] * L
    res = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=A_eq, b_eq=[1], bounds=bounds)
    return -res.fun


class DRCRSSG:
    """DRC-RSSG of Zhang, Chen & Ji [10, Algorithm 1].

    Each step, node i observes the realised attack intensity on its in-edges
    (as [10] assumes; here the true number of adversarial in-neighbours),
    updates the empirical distribution q_i, and applies the optimal
    Stackelberg weights from ``rssg_weights``.
    """

    name = "DRC-RSSG"

    def __init__(self, F, mu=0.05):
        self.F, self.mu = F, mu

    def reset(self, ctx):
        n = ctx["adj"].shape[0]
        self.counts = np.zeros((n, self.F + 1))
        adv = np.zeros(n, dtype=bool)
        adv[ctx["adv"]] = True
        self.intensity = np.minimum(ctx["adj"][:, adv].sum(axis=1), self.F)

    def aggregate(self, i, own, vals, ctx):
        c = self.counts[i]
        q = c / c.sum() if c.sum() > 0 else np.zeros(self.F + 1)
        w, w_self = rssg_weights(np.abs(vals - own), self.F, q, self.mu)
        c[self.intensity[i]] += 1
        return w_self * own + (w * vals).sum()
