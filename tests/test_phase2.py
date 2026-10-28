"""Phase 2 checks: the comparison methods behave as their papers describe."""

import numpy as np
import pytest

from advagent import attacks as A
from advagent import baselines as B
from advagent import dynamics as d
from advagent import graphs as g
from advagent.functions import LocalFunctions

ZERO = lambda t: 0.0


def test_rssg_closed_form_matches_lp():
    rng = np.random.default_rng(0)
    for _ in range(100):
        F = int(rng.integers(1, 4))
        k = int(rng.integers(F + 1, 12))
        dist = rng.random(k)
        q = rng.dirichlet(np.ones(F + 1))
        w, _ = B.rssg_weights(dist, F, q)
        nonB = np.argsort(dist, kind="stable")[: k - F]
        wn = np.sort(w[nonB])[::-1]
        value = wn.sum() - 2 * sum(q[l] * wn[:l].sum() for l in range(F + 1))
        assert value == pytest.approx(B.rssg_lp_value(dist, F, q), abs=1e-9)


def test_q_consensus_reaches_consensus_in_its_own_setting():
    # [7]: pure consensus, states in [0, 1], no faulty nodes.
    rng = np.random.default_rng(0)
    adj = g.random_robust(10, 2, rng)
    r = d.simulate(adj, LocalFunctions(np.zeros(10)), rng.random(10), B.QConsensus(), 300, ZERO)
    assert r.spread[-1] < 1e-3


def test_lf_metropolis_weights_stay_in_the_lf_shield():
    rng = np.random.default_rng(1)
    adj = g.random_robust(15, 3, rng)
    adv = g.random_f_local_set(adj, 1, 3, rng)
    r = d.simulate(adj, LocalFunctions(np.zeros(15)), rng.uniform(0, 10, 15), B.LFMetropolis(1), 300, ZERO,
                   adversaries=adv, attack=A.UniformRandom(-100, 100, byzantine=True))
    lo, hi = r.XR[0].min(), r.XR[0].max()
    assert lo - 1e-12 <= r.XR.min() and r.XR.max() <= hi + 1e-12
    assert r.spread[-1] < 1e-6


def test_drc_rssg_safety_counterexample():
    """DRC-RSSG drops the F values farthest from the node's own value. A
    Byzantine value outside the regular range but nearer than the farthest
    regular neighbour is kept, so one step leaves the safety interval that
    [10, Problem 1, Thm. 1] claims is invariant. K5 is 3-robust, so the
    graph conditions of both [1] and [10] hold."""
    adj = g.complete(5)
    x0 = np.array([10, 9.9, 9.9, 0, 0.0])
    attack = lambda t, x, ctx: np.full((5, 1), 19.0)
    f = LocalFunctions(np.zeros(5))
    rssg = d.simulate(adj, f, x0, B.DRCRSSG(1), 1, ZERO, adversaries=[4], attack=attack)
    assert rssg.XR[1].max() > 12.5
    for rule in (d.LF(1), B.SuVaidya(1)):
        r = d.simulate(adj, f, x0, rule, 1, ZERO, adversaries=[4], attack=attack)
        assert r.XR[1].max() <= 10


@pytest.mark.parametrize("rule", [d.LF(1), B.LFMetropolis(1), B.SuVaidya(1)])
def test_trimming_rules_are_safe_against_the_farthest_evader(rule):
    for seed in range(4):
        rng = np.random.default_rng(seed)
        adj = g.random_robust(20, 3, rng, extra=1)
        adv = g.random_f_local_set(adj, 1, 4, rng)
        x0 = rng.uniform(0, 10, 20)
        r = d.simulate(adj, LocalFunctions(np.zeros(20)), x0, rule, 200, ZERO,
                       adversaries=adv, attack=A.FarthestEvader(+1, 0.9))
        reg0 = x0[r.regular]
        assert reg0.min() - 1e-12 <= r.XR.min() and r.XR.max() <= reg0.max() + 1e-12
