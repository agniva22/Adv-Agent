"""Consensus-based distributed optimization under adversarial nodes.

Each regular node i forms a consensus point y_i from its own value and the
values it receives, then takes a subgradient step [1, eq. (8)]:

    x_i(t+1) = y_i - alpha_t * d_i,   d_i a subgradient of f_i at y_i.

Aggregation rules decide how y_i is formed:
  * DGD: uniform weights over self and all in-neighbours [1, eq. (2)].
  * LF:  drop up to F values above and F below x_i, then uniform weights over
         self and the retained set J_i(t) [1, Sec. V].
"""

from dataclasses import dataclass, field

import numpy as np


def lf_keep(own, vals, F):
    """Mask of neighbour values retained by LF with parameter F.

    Removes the F largest values that are strictly larger than ``own`` (all of
    them if there are fewer than F) and likewise the F smallest that are
    strictly smaller. Ties are broken by position, which is arbitrary as [1]
    allows.
    """
    keep = np.ones(vals.size, dtype=bool)
    above = np.flatnonzero(vals > own)
    if above.size:
        drop = above[np.argsort(-vals[above], kind="stable")[:F]]
        keep[drop] = False
    below = np.flatnonzero(vals < own)
    if below.size:
        drop = below[np.argsort(vals[below], kind="stable")[:F]]
        keep[drop] = False
    return keep


class DGD:
    """Metropolis weights a_ij = 1 / (1 + max(d_i, d_j)): doubly stochastic on
    undirected graphs, as Prop. 3.3 of [1] requires for convergence to x*.
    (Uniform weights would instead converge to the minimizer of a
    centrality-weighted sum, by Lemma 3.4 of [1].)"""

    name = "DGD"

    def aggregate(self, i, own, vals, ctx):
        adj = ctx["adj"]
        deg = adj.sum(axis=1)
        w = 1.0 / (1 + np.maximum(deg[i], deg[adj[i]]))
        return own + (w * (vals - own)).sum()


class LF:
    name = "LF"

    def __init__(self, F):
        self.F = F

    def aggregate(self, i, own, vals, ctx):
        kept = vals[lf_keep(own, vals, self.F)]
        return (own + kept.sum()) / (kept.size + 1)


def harmonic_steps(c=1.0, t0=1.0):
    """alpha_t = c / (t + t0): sum alpha_t = inf and sum alpha_t^2 < inf."""
    return lambda t: c / (t + t0)


def power_steps(c=0.5, p=0.6, t0=1.0):
    """alpha_t = c / (t + t0)^p; for 1/2 < p <= 1 both step-size conditions of [1] hold."""
    return lambda t: c / (t + t0) ** p


@dataclass
class Result:
    X: np.ndarray            # (T+1, n) node states; adversary rows hold what they broadcast
    regular: np.ndarray      # boolean mask of regular nodes
    hull: tuple              # [min, max] of regular nodes' local minimizers
    x_star: float            # minimizer of the regular nodes' average objective
    info: dict = field(default_factory=dict)

    @property
    def XR(self):
        return self.X[:, self.regular]

    @property
    def spread(self):
        """D(t) = M(t) - m(t) over regular nodes [1, Sec. VI-B]."""
        return self.XR.max(axis=1) - self.XR.min(axis=1)

    @property
    def mean(self):
        return self.XR.mean(axis=1)

    def hull_violation(self, burn_in=0):
        """Largest distance by which a regular state lies outside the minimizer hull after ``burn_in``."""
        XR = self.XR[burn_in:]
        lo, hi = self.hull
        return float(max(0.0, lo - XR.min(), XR.max() - hi))

    def error(self):
        """|x_bar_R(T) - x*| at the final step."""
        return float(abs(self.mean[-1] - self.x_star))


def simulate(adj, funcs, x0, rule, T, alpha, adversaries=(), attack=None, seed=0):
    """Run T synchronous steps.

    ``attack(t, X_t, ctx)`` returns what the adversaries send: an array of
    length |A| (malicious: one broadcast value each) or an (n, |A|) array
    (Byzantine: row i is what node i receives from each adversary).
    """
    n = adj.shape[0]
    adv = np.asarray(sorted(adversaries), dtype=int)
    regular = np.ones(n, dtype=bool)
    regular[adv] = False
    reg_idx = np.flatnonzero(regular)
    rng = np.random.default_rng(seed)
    nbrs = [np.flatnonzero(adj[i]) for i in range(n)]
    ctx = {"adj": adj, "adv": adv, "regular": regular, "funcs": funcs, "rng": rng, "rule": rule,
           "nbrs": nbrs, "t": 0}
    if hasattr(rule, "reset"):
        rule.reset(ctx)
    # Gradient at the consensus point y_i as in [1]; Su-Vaidya [5] uses x_i(t).
    grad_at_own = getattr(rule, "grad_at_own", False)

    X = np.empty((T + 1, n))
    X[0] = x0
    for t in range(T):
        ctx["t"] = t
        x = X[t]
        recv = np.broadcast_to(x, (n, n)).copy()  # recv[i, j]: value i gets from j
        if adv.size:
            sent = np.asarray(attack(t, x, ctx), dtype=float)
            recv[:, adv] = sent if sent.ndim == 2 else np.broadcast_to(sent, (n, adv.size))
            X[t + 1, adv] = sent if sent.ndim == 1 else sent.mean(axis=0)
        y = np.array([rule.aggregate(i, x[i], recv[i, nbrs[i]], ctx) for i in reg_idx])
        g_at = x[reg_idx] if grad_at_own else y
        X[t + 1, reg_idx] = y - alpha(t) * funcs.grad(g_at, reg_idx)

    return Result(X, regular, funcs.minimizer_hull(reg_idx), funcs.global_minimizer(reg_idx))
