"""The batched PyTorch simulator matches the NumPy one."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from advagent import attacks as A
from advagent import baselines as B
from advagent import dynamics as d
from advagent import graphs as g
from advagent import torchsim as ts
from advagent.functions import LocalFunctions

T = 200
NP_RULES = {
    "DGD": lambda F: d.DGD(), "LF": lambda F: d.LF(F), "LF-Metropolis": lambda F: B.LFMetropolis(F),
    "Su-Vaidya": lambda F: B.SuVaidya(F), "Trimmed mean": lambda F: B.TrimmedMean(F),
    "IOS": lambda F: B.IOS(F), "ClippedGossip": lambda F: B.ClippedGossip(F), "Q-consensus": lambda F: B.QConsensus(scale=10.0),
    "Q-consensus (fixed beta)": lambda F: B.QConsensus(beta1=0.0, scale=10.0), "DRC-RSSG": lambda F: B.DRCRSSG(F),
}
# (numpy attack factory, torch attack factory)
ATTACKS = {
    "fixed": (lambda F: A.Fixed(50.0), lambda: ts.Fixed(50.0)),
    "evader": (lambda F: A.FarthestEvader(+1, 0.9), lambda: ts.FarthestEvader(+1, 0.9)),
    "trim-hugger": (lambda F: A.TrimHugger(F, +1), lambda: ts.Band(0.99)),
    "switching": (lambda F: A.StallSwitcher(A.TrimHugger(F, +1), A.TrimHugger(F, -1)), lambda: ts.StallSwitcher()),
}


def instances(F, k=3):
    out = []
    for seed in range(k):
        rng = np.random.default_rng(100 + seed)
        adj = g.random_robust(20, 2 * F + 1, rng, extra=1)
        adv = g.random_f_local_set(adj, F, 3 * F, rng)
        out.append({"adj": adj, "adv": adv, "a": rng.uniform(-5, 5, 20), "x0": rng.uniform(-5, 5, 20), "F": F})
    return out


@pytest.mark.parametrize("F", [1, 2])
@pytest.mark.parametrize("method", list(NP_RULES))
@pytest.mark.parametrize("attack", list(ATTACKS))
def test_torch_matches_numpy(F, method, attack):
    insts = instances(F)
    batch = ts.Batch(insts, dtype=torch.float64)
    r = ts.run(batch, ts.BASELINES[method](), T, ts.power_alpha(), ATTACKS[attack][1]())
    for b, I in enumerate(insts):
        np_r = d.simulate(I["adj"], LocalFunctions.quadratic(I["a"], L=100), I["x0"], NP_RULES[method](F), T,
                          d.power_steps(0.5, 0.6), adversaries=I["adv"], attack=ATTACKS[attack][0](F))
        reg = np_r.regular
        np.testing.assert_allclose(r["x"][b].numpy()[reg], np_r.X[-1, reg], rtol=0, atol=1e-6)  # summation order differs
        # NumPy finds x* with a bounded scalar search (accurate to ~1e-8); torch uses the exact mean.
        assert r["error"][b].item() == pytest.approx(np_r.error(), abs=1e-6)
        assert r["oscillation"][b].item() == pytest.approx(np_r.oscillation(T // 2), abs=1e-6)
        assert r["hull_violation"][b].item() == pytest.approx(np_r.hull_violation(T // 2), abs=1e-9)


@pytest.mark.parametrize("method", ["DGD", "LF", "Su-Vaidya", "Trimmed mean"])
def test_torch_stealthy_matches_numpy(method):
    insts = instances(1)
    batch = ts.Batch(ts.stealthy(insts, 5.0), dtype=torch.float64)
    r = ts.run(batch, ts.BASELINES[method](), T, ts.power_alpha())
    for b, I in enumerate(insts):
        np_r = d.simulate(I["adj"], LocalFunctions.quadratic(I["a"], L=100), I["x0"], NP_RULES[method](1), T,
                          d.power_steps(0.5, 0.6), adversaries=I["adv"], attack=A.Stealthy(5.0, d.power_steps(0.5, 0.6)))
        assert r["error"][b].item() == pytest.approx(np_r.error(), abs=1e-6)


def test_band_action_scale():
    """a = +-0.99 hugs the trimming band (kept by LF); a = +-1.5 lies beyond it (dropped)."""
    insts = instances(1)
    batch = ts.Batch(insts, dtype=torch.float64)
    x = batch.x0
    vals = batch.gather(x)
    own = x[..., None]
    for a, kept in [(0.99, True), (-0.99, True), (1.5, False), (-1.5, False)]:
        v, (hi, lo, D) = ts.band(torch.full_like(x, a), x, vals, batch)
        sent = torch.where(batch.from_adv, v[..., None], vals)
        keep = ts.lf_keep(own, sent, batch.nmask, batch.F)
        # receivers where the band is non-degenerate on the pushed side
        side = (hi > x) if a > 0 else (lo < x)
        rows = batch.reg & (batch.n_adv_in > 0) & side
        got = (keep & batch.from_adv).any(-1)[rows]
        assert bool(got.all()) if kept else not bool(got.any())
