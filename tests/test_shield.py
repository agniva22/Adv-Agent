"""The shield keeps [1]'s guarantees for arbitrary policies."""

import pytest

torch = pytest.importorskip("torch")

from advagent import learn as Lr
from advagent import torchsim as ts


def wild_policy(seed, shield=True, scale=5.0):
    """A defender with large random parameters: weights far from LF's."""
    torch.manual_seed(seed)
    pol = Lr.ShieldedDefender(shield=shield)
    with torch.no_grad():
        for p in pol.parameters():
            p.normal_(0, scale)
    return pol


ATTACKS = {
    "fixed": lambda: ts.Fixed(50.0),
    "random": lambda: ts.UniformRandom(-50, 50),
    "trim-hugger": lambda: ts.Band(0.99),
    "switching": lambda: ts.StallSwitcher(),
}


@pytest.mark.parametrize("F", [1, 2])
@pytest.mark.parametrize("attack", list(ATTACKS))
def test_shielded_weights_are_a_valid_lf_convex_combination(F, attack):
    batch = ts.Batch([Lr.instance(500 + s, F) for s in range(4)], dtype=torch.float64)
    pol = wild_policy(F).double()
    x = batch.x0
    vals = batch.gather(x)
    av, _ = ATTACKS[attack]()(0, x, vals, batch, ATTACKS[attack]().init(batch))
    vals = torch.where(batch.from_adv, av, vals)
    w, mask = pol.weights(x, vals, batch, 0.5)
    keep = ts.lf_keep(x[..., None], vals, batch.nmask, batch.F)
    eta = 0.25 / (batch.deg + 1)
    assert torch.allclose(w.sum(-1), torch.ones_like(x))
    assert bool((w[..., 1:][~keep] == 0).all())                       # support: {i} and J_i only
    assert bool((w >= eta[..., None] - 1e-12)[mask].all())             # floor
    assert bool((w[..., 0] >= eta - 1e-12).all())


@pytest.mark.parametrize("F", [1, 2])
@pytest.mark.parametrize("attack", list(ATTACKS))
def test_wild_shielded_policy_keeps_hull_and_consensus(F, attack):
    """Thms. 6.1 and 7.1 for policies nothing like LF ((2F+1)-robust graphs, F-local Byzantine)."""
    batch = ts.Batch([Lr.instance(600 + s, F) for s in range(6)], dtype=torch.float64)
    for seed in range(3):
        with torch.no_grad():
            r = ts.run(batch, wild_policy(seed).double(), 2000, ts.power_alpha(), ATTACKS[attack]())
            excess = ts.range_excess(batch, wild_policy(seed).double(), ATTACKS[attack]())
        assert float(r["hull_violation"].max()) < 1e-9     # Thm. 7.1 is asymptotic: checked on t >= T/2
        assert float(excess.max()) < 1e-9                  # alpha = 0: never leaves the initial range
        assert float(r["spread"].max()) < 1.0


def test_unshielded_policy_can_be_captured():
    """The ablation without the shield has no such guarantee."""
    batch = ts.Batch([Lr.instance(700 + s, 1) for s in range(6)], dtype=torch.float64)
    with torch.no_grad():
        r = ts.run(batch, Lr.ShieldedDefender(shield=False).double(), 2000, ts.power_alpha(), ts.Fixed(50.0))
    assert float(r["hull_violation"].max()) > 1.0


def wild_trust(seed, scale=5.0):
    """Memory + fallback defender with large random parameters (c = 0.05, two memory timescales)."""
    torch.manual_seed(seed)
    pol = Lr.TrustDefender(c=0.05, betas=(0.95, 0.995))
    with torch.no_grad():
        for p in pol.parameters():
            p.normal_(0, scale)
    return pol


@pytest.mark.parametrize("F", [1, 2])
@pytest.mark.parametrize("attack", list(ATTACKS))
def test_memory_fallback_weights_are_valid(F, attack):
    """Memory inputs and the LF blend keep support, floor and sum (shield_theorem.md, Remarks)."""
    batch = ts.Batch([Lr.instance(500 + s, F) for s in range(4)], dtype=torch.float64)
    pol = wild_trust(F).double()
    pol.reset(batch)
    att = ATTACKS[attack]()
    state = att.init(batch)
    x = batch.x0
    eta = 0.05 / (batch.deg + 1)
    for t in range(3):   # t >= 1 uses the memory written at t - 1
        vals = batch.gather(x)
        av, state = att(t, x, vals, batch, state)
        vals = torch.where(batch.from_adv, av, vals)
        w, mask = pol.weights(x, vals, batch, 0.5, t)
        keep = ts.lf_keep(x[..., None], vals, batch.nmask, batch.F)
        assert torch.allclose(w.sum(-1), torch.ones_like(x))
        assert bool((w[..., 1:][~keep] == 0).all())
        assert bool((w >= eta[..., None] - 1e-12)[mask].all())
        x = (w * torch.cat([x[..., None], vals], -1)).sum(-1)


@pytest.mark.parametrize("attack", list(ATTACKS))
def test_wild_memory_fallback_keeps_hull_and_consensus(attack):
    batch = ts.Batch([Lr.instance(600 + s, 1) for s in range(6)], dtype=torch.float64)
    for seed in range(2):
        with torch.no_grad():
            r = ts.run(batch, wild_trust(seed).double(), 2000, ts.power_alpha(), ATTACKS[attack]())
            excess = ts.range_excess(batch, wild_trust(seed).double(), ATTACKS[attack]())
        assert float(r["hull_violation"].max()) < 1e-9
        assert float(excess.max()) < 1e-9
        assert float(r["spread"].max()) < 1.0
