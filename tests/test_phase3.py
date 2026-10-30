"""Phase 3 checks: the new scripted attacks do what their docstrings claim."""

import numpy as np
import pytest

from advagent import attacks as A
from advagent import baselines as B
from advagent import dynamics as d
from advagent import graphs as g
from advagent.functions import LocalFunctions

ALPHA = d.power_steps(0.5, 0.6)


def _instance(seed, F=1, size=3):
    rng = np.random.default_rng(seed)
    adj = g.random_robust(20, 2 * F + 1, rng, extra=1)
    adv = g.random_f_local_set(adj, F, size, rng)
    a = rng.uniform(-5, 5, 20)
    return adj, adv, a, rng.uniform(-5, 5, 20)


@pytest.mark.parametrize("rule", [d.DGD, lambda: d.LF(1), lambda: B.SuVaidya(1)])
def test_stealthy_is_indistinguishable_from_a_regular_node(rule):
    """[1, Thm. 4.4]: the regular nodes see exactly what they would see if the
    adversaries were honest nodes whose objective has minimizer ``fake``."""
    adj, adv, a, x0 = _instance(0)
    fake = 3.0
    attacked = d.simulate(adj, LocalFunctions.quadratic(a, L=100), x0, rule(), 300, ALPHA,
                          adversaries=adv, attack=A.Stealthy(fake, ALPHA))
    honest_a = a.copy()
    honest_a[adv] = fake
    honest = d.simulate(adj, LocalFunctions.quadratic(honest_a, L=100), x0, rule(), 300, ALPHA)
    np.testing.assert_allclose(attacked.XR, honest.X[:, attacked.regular], atol=1e-12)


@pytest.mark.parametrize("F", [1, 2])
def test_trim_hugger_values_are_kept_by_lf(F):
    adj, adv, a, x0 = _instance(1, F, 3 * F)
    regular = np.ones(20, dtype=bool)
    regular[adv] = False
    ctx = {"adj": adj, "regular": regular, "adv": np.asarray(adv)}
    sent = A.TrimHugger(F, +1)(0, x0, ctx)
    hugged = 0
    for i in np.flatnonzero(regular):
        nb = np.flatnonzero(adj[i])
        vals = x0[nb].copy()
        is_adv = np.isin(nb, adv)
        vals[is_adv] = sent[i, 0]
        keep = d.lf_keep(x0[i], vals, F)
        if sent[i, 0] > x0[i]:
            hugged += 1
            assert keep[is_adv].all()
    assert hugged > 0


def test_switching_keeps_lf_oscillating_but_safe():
    adj, adv, a, x0 = _instance(0)
    att = A.StallSwitcher(A.TrimHugger(1, +1), A.TrimHugger(1, -1))
    f = LocalFunctions.quadratic(a, L=100)
    r = d.simulate(adj, f, x0, d.LF(1), 2000, ALPHA, adversaries=adv, attack=att)
    assert len(att.switches) >= 4
    assert r.oscillation(1000) > 0.5
    assert r.hull_violation(10) == 0.0
