"""Phase 1 checks: the simulator reproduces the claims of Sundaram & Gharesifard [1]."""

import networkx as nx
import numpy as np
import pytest

from advagent import attacks as A
from advagent import dynamics as d
from advagent import graphs as g
from advagent.functions import ABS, LocalFunctions


# --- graph claims ---------------------------------------------------------

def test_fig1_graph_is_2_robust_and_22_robust_but_not_3_robust():
    adj, _ = g.fig1_graph()
    assert g.max_robustness(adj) == 2
    assert g.is_rs_robust(adj, 2, 2)


def test_uw_graph_is_3_robust_with_U_the_maximum_1_local_set():
    adj, W, U = g.uw_graph(2)
    assert g.max_robustness(adj) == 3
    S = g.max_f_local_set(adj, 1)
    assert set(np.flatnonzero(S)) == set(U)


@pytest.mark.parametrize("n,F", [(5, 1), (8, 2)])
def test_max_f_local_set_of_complete_graph_has_size_F(n, F):
    assert g.max_f_local_set(g.complete(n), F).sum() == F


@pytest.mark.parametrize("r", [2, 3])
def test_random_robust_generator(r):
    adj = g.random_robust(10, r, np.random.default_rng(r))
    assert g.max_robustness(adj) >= r


# --- LF filter ---------------------------------------------------------------

def test_lf_keep_drops_only_values_beyond_own():
    vals = np.array([5.0, -3.0, 1.0, 2.0, -1.0])
    keep = d.lf_keep(0.0, vals, 1)
    assert keep.tolist() == [False, False, True, True, True]
    # Fewer than F values above own: all of them are dropped.
    assert d.lf_keep(10.0, vals, 1).tolist() == [True, False, True, True, True]


# --- dynamics claims ---------------------------------------------------------

def test_prop_4_3_dgd_is_captured_but_lf_is_not():
    rng = np.random.default_rng(1)
    n = 10
    adj = g.random_robust(n, 3, rng)
    a = rng.uniform(-1, 1, n)
    f = LocalFunctions.quadratic(a)
    kw = dict(T=3000, alpha=d.power_steps(0.5, 0.6), adversaries=[n - 1], attack=A.Fixed(50.0))
    dgd = d.simulate(adj, f, a.copy(), d.DGD(), **kw)
    lf = d.simulate(adj, f, a.copy(), d.LF(1), **kw)
    assert dgd.mean[-1] > 40
    assert lf.hull_violation(burn_in=10) == 0


def test_dgd_is_exact_without_adversaries():
    adj, W, U = g.uw_graph(3)
    a = np.zeros(adj.shape[0])
    a[U] = 4.0
    r = d.simulate(adj, LocalFunctions.quadratic(a, L=100), a.copy(), d.DGD(), 4000, d.power_steps(0.5, 0.6))
    assert r.error() < 1e-3


def test_ex_6_2_filtered_graph_not_rooted_yet_consensus_under_any_single_adversary():
    adj, x0 = g.fig1_graph()
    G = nx.DiGraph()
    G.add_nodes_from(range(5))
    for i in range(5):
        nb = np.flatnonzero(adj[i])
        G.add_edges_from((j, i) for j in nb[d.lf_keep(x0[i], x0[nb], 1)])
    assert not any(len(nx.descendants(G, v)) == 4 for v in G)

    f = LocalFunctions(np.zeros(5))
    for m in range(5):
        for att in (A.Fixed(5.0), A.Fixed(-5.0), A.UniformRandom(-10, 10), A.UniformRandom(-10, 10, byzantine=True)):
            r = d.simulate(adj, f, x0, d.LF(1), 400, lambda t: 0.0, adversaries=[m], attack=att, seed=m)
            assert r.spread[-1] < 1e-9
            lo, hi = x0[r.regular].min(), x0[r.regular].max()
            assert lo - 1e-12 <= r.XR.min() and r.XR.max() <= hi + 1e-12


def test_ex_7_2_switching_attack_prevents_convergence_but_stays_in_hull():
    adj = g.complete(5)
    f = LocalFunctions.quadratic([0, 0, 0, 9, 0], L=100)
    att = A.Switching(low=[0, 1, 2], odd=3)
    T = 3000
    r = d.simulate(adj, f, np.array([0, 0, 0, 9, 0.0]), d.LF(1), T, d.power_steps(0.5, 0.6),
                   adversaries=[4], attack=att)
    tail = r.mean[T // 2:]
    assert len(att.switches) >= 10
    swing = tail.max() - tail.min()
    assert swing > 1.5                                # still oscillating late in the run
    assert r.spread[T // 2:].max() < 0.1 * swing      # ... with the nodes moving together
    assert r.hull_violation(burn_in=10) == 0


@pytest.mark.parametrize("K", [2, 5])
def test_ex_8_2_lf_error_is_b_over_4(K):
    b = 4.0
    adj, W, U = g.uw_graph(K)
    a = np.zeros(adj.shape[0])
    a[U] = b
    r = d.simulate(adj, LocalFunctions.quadratic(a, L=100), a.copy(), d.LF(1), 4000, d.power_steps(0.5, 0.6))
    assert r.x_star == pytest.approx(b / 4, abs=1e-6)
    assert r.error() == pytest.approx(b / 4, abs=0.05)


@pytest.mark.parametrize("F", [1, 2])
def test_thm_6_1_and_7_1_f_local_byzantine_on_2f_plus_1_robust_graphs(F):
    rng = np.random.default_rng(10 + F)
    for trial in range(5):
        n = 14
        adj = g.random_robust(n, 2 * F + 1, rng, extra=1)
        adv = g.random_f_local_set(adj, F, size=3 * F, rng=rng)
        a = rng.uniform(-2, 2, n)
        kind = np.where(rng.random(n) < 0.3, ABS, 0)
        f = LocalFunctions(a, kind=kind, L=50)
        r = d.simulate(adj, f, rng.uniform(-20, 20, n), d.LF(F), 3000, d.power_steps(0.5, 0.6),
                       adversaries=adv, attack=A.UniformRandom(-100, 100, byzantine=True), seed=trial)
        assert r.spread[-1] < 0.05
        assert r.hull_violation(burn_in=2000) < 0.05
