"""Graph construction and graph-theoretic checks from Sundaram & Gharesifard [1].

Convention: ``adj[i, j] = True`` means edge (v_j, v_i), i.e. j is an in-neighbour
of i and i receives j's value. Undirected graphs have symmetric ``adj``.
"""

from itertools import product

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def from_edges(n, edges, undirected=True):
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[v, u] = True
        if undirected:
            adj[u, v] = True
    np.fill_diagonal(adj, False)
    return adj


def complete(n):
    adj = np.ones((n, n), dtype=bool)
    np.fill_diagonal(adj, False)
    return adj


def fig1_graph():
    """5-node graph of [1, Fig. 1(a)] / Example 6.2.

    Node order: 0 = top (initial value 2), 1, 2 = middle (value 0),
    3, 4 = bottom (value 1). Returns (adj, initial values).
    """
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 4)]
    return from_edges(5, edges), np.array([2.0, 0.0, 0.0, 1.0, 1.0])


def uw_graph(K):
    """Graph of [1, Fig. 3] / Example 8.2.

    W = {0..3K-1} is complete; U = {3K..4K-1}, and u_k connects to three
    unique vertices of W. Returns (adj, indices of W, indices of U).
    """
    N = 3 * K
    edges = [(a, b) for a in range(N) for b in range(a + 1, N)]
    for k in range(K):
        edges += [(N + k, 3 * k + m) for m in range(3)]
    W, U = np.arange(N), np.arange(N, N + K)
    return from_edges(4 * K, edges), W, U


def random_robust(n, r, rng, extra=0):
    """Random undirected r-robust graph.

    Start from K_{2r-1}, which is r-robust, and add nodes one at a time with
    edges to ``r + extra`` distinct existing nodes. Adding a node with at least
    r in-neighbours preserves r-robustness [2].
    """
    m0 = max(2 * r - 1, r + extra + 1)
    if n < m0:
        raise ValueError(f"need n >= {m0}")
    adj = np.zeros((n, n), dtype=bool)
    adj[:m0, :m0] = True
    for v in range(m0, n):
        nbrs = rng.choice(v, size=r + extra, replace=False)
        adj[v, nbrs] = adj[nbrs, v] = True
    np.fill_diagonal(adj, False)
    return adj


def erdos_renyi(n, p, rng):
    upper = np.triu(rng.random((n, n)) < p, 1)
    return upper | upper.T


def _subset_pairs(n):
    """All pairs (S1, S2) of disjoint nonempty subsets, as boolean masks.

    Each node is labelled 0 (in neither), 1 (in S1) or 2 (in S2); pairs are
    unordered, so only labellings with S1's smallest member before S2's are kept.
    """
    for labels in product((0, 1, 2), repeat=n):
        lab = np.array(labels)
        s1, s2 = lab == 1, lab == 2
        if s1.any() and s2.any() and np.argmax(s1) < np.argmax(s2):
            yield s1, s2


def _outside_counts(adj, S):
    """For each node, the number of in-neighbours outside S."""
    return adj[:, ~S].sum(axis=1)


def is_rs_robust(adj, r, s):
    """Brute-force check of (r, s)-robustness [1, Def. 2.2]; feasible for n <= ~12."""
    n = adj.shape[0]
    for S1, S2 in _subset_pairs(n):
        c1 = _outside_counts(adj, S1)[S1] >= r
        c2 = _outside_counts(adj, S2)[S2] >= r
        if not (c1.all() or c2.all() or c1.sum() + c2.sum() >= s):
            return False
    return True


def is_r_robust(adj, r):
    """r-robustness is (r, 1)-robustness [1, Def. 2.1]."""
    return is_rs_robust(adj, r, 1)


def max_robustness(adj):
    """Largest r for which the graph is r-robust (brute force)."""
    r = 0
    while r + 1 <= adj.shape[0] and is_r_robust(adj, r + 1):
        r += 1
    return r


def is_f_local(adj, S, F):
    """S is F-local if every node outside S has at most F in-neighbours in S."""
    S = np.asarray(S, dtype=bool)
    return bool((adj[~S][:, S].sum(axis=1) <= F).all())


def random_f_local_set(adj, F, size, rng):
    """A random F-local adversary set of at most ``size`` nodes (greedy in random order)."""
    S = np.zeros(adj.shape[0], dtype=bool)
    for v in rng.permutation(adj.shape[0]):
        if S.sum() == size:
            break
        S[v] = True
        if not is_f_local(adj, S, F):
            S[v] = False
    return np.flatnonzero(S)


def max_f_local_set(adj, F):
    """Maximum F-local set via a 0/1 integer program (the problem is NP-hard [1, Thm. 8.4]).

    For each node v: sum_{u in N(v)} z_u <= F + deg(v) * z_v, i.e. the
    constraint is active only when v is outside the set. The set must leave at
    least one node outside (S = V is vacuously F-local); with this, the
    maximum F-local set of a complete graph has size F, as stated in [1].
    """
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    A = adj.astype(float) - np.diag(deg.astype(float))
    res = milp(
        c=-np.ones(n),
        constraints=[LinearConstraint(A, -np.inf, F), LinearConstraint(np.ones((1, n)), 0, n - 1)],
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.x > 0.5
