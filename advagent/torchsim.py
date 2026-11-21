"""Batched PyTorch version of ``dynamics.simulate`` for learning (Phases 4-7).

B episodes run in lockstep, one graph each (all with the same number of nodes
n). Neighbour lists are padded to the largest in-degree d: ``nbr[b, i, k]`` is
the k-th in-neighbour of node i in episode b and ``nmask`` marks real slots.
Everything is a tensor op, so a run is fast on a GPU and differentiable with
respect to a learned defender's weights.

Conventions match ``dynamics.simulate``: each regular node forms y_i from its
own value and the received values, then x_i <- y_i - alpha_t * grad f_i(y_i)
(at x_i for Su-Vaidya). Objectives are capped quadratics (x - a_i)^2.
``test_phase4.py`` checks every rule here against its NumPy counterpart.

Attacks are functional so that gradient checkpointing can replay them:
``state = attack.init(batch)`` and ``values, state = attack(t, x, vals, batch, state)``,
where ``values[b, i, k]`` is what node i receives on slot k (used only on
slots whose sender is an adversary). Attack states are tuples of tensors.
"""

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

S = 10.0     # state scale used to normalise policy features
VMAX = 50.0  # bound on the values a band-parametrised attacker can send


class Batch:
    """B problem instances as padded tensors.

    Each instance is a dict with ``adj`` (n x n bool, adj[i, j] means j -> i),
    ``adv`` (adversary indices), ``a`` (minimizers), ``x0`` and optionally
    ``eval`` (mask of the nodes the metrics are over; defaults to the
    non-adversaries) and ``F``.
    """

    def __init__(self, instances, F=None, device="cpu", dtype=torch.float32, L=100.0):
        B, n = len(instances), instances[0]["adj"].shape[0]
        adj = np.stack([I["adj"] for I in instances])
        deg = adj.sum(2)
        d = int(deg.max())
        nbr = np.repeat(np.arange(n)[None, :, None], B, 0).repeat(d, 2)
        nmask = np.zeros((B, n, d), dtype=bool)
        for b in range(B):
            for i in range(n):
                js = np.flatnonzero(adj[b, i])
                nbr[b, i, : js.size] = js
                nmask[b, i, : js.size] = True
        adv = np.zeros((B, n), dtype=bool)
        for b, I in enumerate(instances):
            adv[b, list(I["adv"])] = True
        ev = np.stack([I["eval"] if I.get("eval") is not None else ~adv[b] for b, I in enumerate(instances)])
        Fs = np.array([I.get("F", F) for I in instances])

        T = lambda arr, dt=dtype: torch.as_tensor(arr, device=device, dtype=dt)
        self.B, self.n, self.d, self.L = B, n, d, L
        self.device, self.dtype = device, dtype
        self.nbr = T(nbr, torch.long)
        self.nmask = T(nmask, torch.bool)
        self.adv = T(adv, torch.bool)
        self.reg = ~self.adv
        self.ev = T(ev, torch.bool)
        self.F = T(Fs, torch.long).view(B, 1, 1)
        self.from_adv = self.nmask & self.gather(self.adv)
        self.n_adv_in = self.from_adv.sum(-1)
        self.a = T(np.stack([I["a"] for I in instances]))
        # Optional curvature s_i: f_i = s_i (x - a_i)^2 (default 1, the plain quadratics).
        self.s = T(np.stack([np.asarray(I.get("s", np.ones(n)), dtype=float) for I in instances]))
        self.x0 = T(np.stack([I["x0"] for I in instances]))
        self.deg = T(deg)
        self.nbr_deg = self.gather(self.deg)
        evf = self.ev.to(dtype)
        # x* of the average of capped quadratics: the s-weighted mean while no cap is active.
        self.x_star = (self.s * self.a * evf).sum(1) / (self.s * evf).sum(1)
        big = torch.finfo(dtype).max
        self.hull_lo = torch.where(self.ev, self.a, big).min(1).values
        self.hull_hi = torch.where(self.ev, -self.a, big).min(1).values.neg()
        assert float((self.a - self.x_star[:, None]).abs().where(self.ev, 0).max()) < L / 2

    def gather(self, v):
        """v (B, n) -> values at each node's in-neighbour slots, (B, n, d)."""
        return v.gather(1, self.nbr.view(self.B, -1)).view(self.B, self.n, self.d)

    def mean(self, x):
        """Mean over the evaluated (regular) nodes, per episode."""
        evf = self.ev.to(x.dtype)
        return (x * evf).sum(1) / evf.sum(1)

    def grad(self, x):
        return (2 * self.s * (x - self.a)).clamp(-self.L, self.L)


# --------------------------------------------------------------------- rules

def _rank(key):
    """Position of each entry in a stable ascending sort along the last axis."""
    return key.sort(dim=-1, stable=True).indices.argsort(-1)


def lf_keep(own, vals, nmask, F):
    """Batched ``dynamics.lf_keep``: own (B, n, 1), vals (B, n, d), F (B, 1, 1)."""
    above = nmask & (vals > own)
    below = nmask & (vals < own)
    inf = torch.tensor(float("inf"), device=vals.device, dtype=vals.dtype)
    ru = _rank(torch.where(above, -vals, inf))
    rd = _rank(torch.where(below, vals, inf))
    return nmask & ~((above & (ru < F)) | (below & (rd < F)))


class Rule:
    name = "rule"
    grad_at_own = False

    def reset(self, batch):
        pass

    def aggregate(self, x, vals, batch, t, at):
        raise NotImplementedError


def _weighted(own, vals, w, w_self):
    return w_self * own + (w * vals).sum(-1)


class LF(Rule):
    name = "LF"

    def aggregate(self, x, vals, batch, t, at):
        keep = lf_keep(x[..., None], vals, batch.nmask, batch.F).to(x.dtype)
        return (x + (keep * vals).sum(-1)) / (1 + keep.sum(-1))


class DGD(Rule):
    name = "DGD"

    def aggregate(self, x, vals, batch, t, at):
        w = batch.nmask / (1 + torch.maximum(batch.deg[..., None], batch.nbr_deg))
        return x + (w * (vals - x[..., None])).sum(-1)


class LFMetropolis(Rule):
    name = "LF-Metropolis"

    def aggregate(self, x, vals, batch, t, at):
        keep = lf_keep(x[..., None], vals, batch.nmask, batch.F)
        w = keep / (1 + torch.maximum(batch.deg[..., None], batch.nbr_deg))
        return x + (w * (vals - x[..., None])).sum(-1)


class TrimmedMean(Rule):
    """Batched ``baselines.TrimmedMean`` (the stand-in for Su-Vaidya until 2026-11-03)."""

    name = "Trimmed mean"
    grad_at_own = True

    def aggregate(self, x, vals, batch, t, at):
        inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        pool = torch.cat([x[..., None], torch.where(batch.nmask, vals, inf)], -1).sort(-1).values
        m = batch.deg[..., None] + 1                       # pool size per node
        k = torch.arange(pool.shape[-1], device=x.device)
        F = batch.F.to(x.dtype)
        mid = (k >= F) & (k < m - F)
        cnt = mid.sum(-1)
        y = torch.where(mid, pool, 0).sum(-1) / cnt.clamp(min=1)
        return torch.where(cnt > 0, y, x)


class SuVaidya(Rule):
    """Batched ``baselines.SuVaidya``: [5, Alg. 1] with gradient exchange."""

    name = "Su-Vaidya"
    exchanges_gradients = True

    def update(self, x, vals, g, gvals, batch, t, at):
        inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        F, deg = batch.F, batch.deg[..., None]
        k = torch.arange(batch.d + 1, device=x.device)
        rv = torch.where(batch.nmask, vals, inf).sort(-1).values          # received states, padding last
        mid = (k[: batch.d] >= F) & (k[: batch.d] < deg - F)
        xt = (x + torch.where(mid, rv, 0).sum(-1)) / (mid.sum(-1) + 1)
        gp = torch.cat([g[..., None], torch.where(batch.nmask, gvals, inf)], -1).sort(-1).values
        lo = gp.gather(-1, F.expand(-1, batch.n, 1))[..., 0]               # min of the trimmed pool
        hi = gp.gather(-1, (deg.long() - F).clamp(min=0))[..., 0]          # max: index deg + 1 - F - 1
        xn = (xt - at * 0.5 * (lo + hi)).clamp(-VMAX, VMAX)
        return torch.where(batch.deg >= 2 * F[..., 0] + 1, xn, x)


def half_steps(x, vals, g, gvals, batch, at):
    """Own and received half-step models x_j - alpha g_j(x_j) for rules that run the
    local step before communicating (ClippedGossip, IOS). Adversary slots carry
    whatever the adversary sent."""
    return x - at * g, torch.where(batch.from_adv, vals, vals - at * gvals)


def metropolis(batch):
    """Metropolis-Hastings weights on the in-neighbour slots and the self-weight."""
    w = batch.nmask / (1 + torch.maximum(batch.deg[..., None], batch.nbr_deg))
    return w, 1 - w.sum(-1)


class IOS(Rule):
    """Iterative outlier scissor of Wu et al. [arXiv 2206.04568, Alg. 2], q_n = F:
    F times, drop the neighbour farthest from the weighted average of the trusted
    set (the node itself is never dropped); output the weighted average of the
    rest. Metropolis weights; gradient step first, then aggregation of half-steps."""

    name = "IOS"
    exchanges_gradients = True

    def update(self, x, vals, g, gvals, batch, t, at):
        h, hv = half_steps(x, vals, g, gvals, batch, at)
        w, ws = metropolis(batch)
        keep = batch.nmask.clone()
        for i in range(int(batch.F.max())):
            kw = torch.where(keep, w, 0)
            avg = (ws * h + (kw * hv).sum(-1)) / (ws + kw.sum(-1))
            dist = torch.where(keep, (hv - avg[..., None]).abs(), -1.0)
            drop = torch.zeros_like(keep).scatter_(-1, dist.argmax(-1, keepdim=True), True)
            keep = keep & ~(drop & (i < batch.F) & (dist >= 0))
        kw = torch.where(keep, w, 0)
        return (ws * h + (kw * hv).sum(-1)) / (ws + kw.sum(-1))


class ClippedGossip(Rule):
    """Self-centred clipping of He, Karimireddy and Jaggi [arXiv 2202.01545]:
    x_i <- h_i + sum_j W_ij clip(h_j - h_i, tau_i) on the half-steps h, with
    Metropolis weights and the adaptive radius of their App. D.5:
    tau_i^2 = sum over the closest neighbours S of W_ij |h_i - h_j|^2, where S is
    as large as possible with sum_S W_ij <= 1 - delta_i. delta_i (the Byzantine
    weight) is bounded by the sum of node i's F largest neighbour weights, which
    i can compute. Deterministic gradients, so no momentum."""

    name = "ClippedGossip"
    exchanges_gradients = True

    def update(self, x, vals, g, gvals, batch, t, at):
        h, hv = half_steps(x, vals, g, gvals, batch, at)
        w, _ = metropolis(batch)
        diff = torch.where(batch.nmask, hv - h[..., None], 0)
        idx = (batch.F - 1).expand(-1, batch.n, 1)
        delta = torch.where(batch.nmask, w, 0).sort(-1, descending=True).values.cumsum(-1).gather(-1, idx)
        inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        order = torch.where(batch.nmask, diff.abs(), inf).sort(dim=-1, stable=True).indices
        ws, ds = w.gather(-1, order), diff.gather(-1, order)
        inS = batch.nmask.gather(-1, order) & (ws.cumsum(-1) <= 1 - delta + 1e-12)
        tau = torch.where(inS, ws * ds ** 2, 0).sum(-1).sqrt()
        return h + (w * diff.clamp(-tau[..., None], tau[..., None])).sum(-1)


class QConsensus(Rule):
    """Batched ``baselines.QConsensus``; the credibility matrix is the rule's state."""

    def __init__(self, beta0=10.0, beta1=0.1, eta=0.01, scale=10.0, name="Q-consensus"):
        self.beta0, self.beta1, self.eta, self.scale, self.name = beta0, beta1, eta, scale, name

    def reset(self, batch):
        self.Q = torch.ones_like(batch.nmask, dtype=batch.dtype)

    def aggregate(self, x, vals, batch, t, at):
        beta = self.beta0 + self.beta1 * t
        r = torch.exp(-beta * (vals - x[..., None]).abs() / self.scale)
        self.Q = torch.where(batch.nmask, self.Q + self.eta * (r - self.Q), self.Q)
        q = self.Q * batch.nmask
        self_w = 1 / (batch.deg + 1)
        qs = q.sum(-1, keepdim=True)
        w = torch.where(qs > 0, q / qs.clamp(min=1e-300), batch.nmask / batch.deg[..., None]) * (1 - self_w[..., None])
        return _weighted(x, vals, w, self_w)


class DRCRSSG(Rule):
    """Batched ``baselines.DRCRSSG``. The observed attack intensity is the same
    every step, so the empirical distribution is empty at t = 0 and a point
    mass at min(#adversarial in-neighbours, F) afterwards."""

    name = "DRC-RSSG"

    def __init__(self, mu=0.05):
        self.mu = mu

    def aggregate(self, x, vals, batch, t, at):
        dist = torch.where(batch.nmask, (vals - x[..., None]).abs(), -1.0)
        rank = _rank(-dist)
        nonB = batch.nmask & (rank >= batch.F)
        K = (batch.deg - batch.F[..., 0]).clamp(min=0)
        l = torch.minimum(batch.n_adv_in, batch.F[..., 0]).to(x.dtype) if t > 0 else torch.zeros_like(x)
        gain = 1 - 2 * torch.minimum(l, K) / K.clamp(min=1)
        mu = torch.minimum(torch.full_like(x, self.mu), 1 / (K + 1))
        edge = torch.where(gain > 0, (1 - mu) / K.clamp(min=1), mu)
        w = nonB * edge[..., None]
        self_w = torch.where(gain > 0, mu, 1 - K * mu)
        y = _weighted(x, vals, w, self_w)
        return torch.where(K > 0, y, x)


BASELINES = {
    "DGD": DGD, "LF": LF, "LF-Metropolis": LFMetropolis, "Su-Vaidya": SuVaidya, "Trimmed mean": TrimmedMean, "IOS": IOS, "ClippedGossip": ClippedGossip,
    "Q-consensus": QConsensus,
    "Q-consensus (fixed beta)": lambda: QConsensus(beta1=0.0, name="Q-consensus (fixed beta)"),
    "DRC-RSSG": DRCRSSG,
}


# ------------------------------------------------------------------- attacks

def band(a, x, vals, batch):
    """Map a per-receiver action a (B, n) to the value it sends.

    The scale is the receiver's trimming band: a = 0 sends x_i, a = +1 the
    F-th largest regular in-neighbour value above x_i (x_i if there are fewer
    than F), a = -1 the F-th smallest below; beyond +-1 the value moves further
    out by a multiple of D_i, the distance to the farthest regular in-neighbour.
    |a| < 1 is kept by LF; |a| > 1 is (generically) trimmed. Values are
    clipped to [-VMAX, VMAX], the range the scripted attacks use; without a
    bound a learned attacker can drive DGD's states to overflow.
    """
    own = x[..., None]
    regslot = batch.nmask & ~batch.from_adv
    inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    up = torch.where(regslot & (vals > own), vals, -inf).sort(-1, descending=True).values
    dn = torch.where(regslot & (vals < own), vals, inf).sort(-1).values
    idx = (batch.F - 1).expand(-1, batch.n, 1)
    hi, lo = up.gather(-1, idx)[..., 0], dn.gather(-1, idx)[..., 0]
    hi = torch.where(torch.isfinite(hi), hi, x)
    lo = torch.where(torch.isfinite(lo), lo, x)
    D = torch.where(regslot, (vals - own).abs(), 0).amax(-1).clamp(min=1e-6)
    v = (x + a.clamp(0, 1) * (hi - x) + (a - 1).clamp(min=0) * D
         + a.clamp(-1, 0) * (x - lo) + (a + 1).clamp(max=0) * D)
    return v.clamp(-VMAX, VMAX), (hi, lo, D)


class Attack:
    name = "attack"

    def init(self, batch):
        return ()

    def __call__(self, t, x, vals, batch, state):
        raise NotImplementedError


class Fixed(Attack):
    def __init__(self, value):
        self.value, self.name = value, f"fixed {value:+g}"

    def __call__(self, t, x, vals, batch, state):
        return torch.full_like(vals, self.value), state


class UniformRandom(Attack):
    """Byzantine: a fresh uniform value per edge per step."""

    def __init__(self, lo, hi):
        self.lo, self.hi, self.name = lo, hi, "random Byzantine"

    def __call__(self, t, x, vals, batch, state):
        return self.lo + (self.hi - self.lo) * torch.rand_like(vals), state


class FarthestEvader(Attack):
    def __init__(self, direction=1.0, frac=0.9):
        self.direction, self.frac, self.name = direction, frac, "farthest evader"

    def __call__(self, t, x, vals, batch, state):
        regslot = batch.nmask & ~batch.from_adv
        D = torch.where(regslot, (vals - x[..., None]).abs(), 0).amax(-1)
        return (x + self.direction * self.frac * D)[..., None].expand_as(vals), state


class Band(Attack):
    """Constant band action; a = 0.99 is ``attacks.TrimHugger(F, +1)``."""

    def __init__(self, a, name=None):
        self.a, self.name = a, name or f"band {a:+g}"

    def __call__(self, t, x, vals, batch, state):
        v, _ = band(torch.full_like(x, self.a), x, vals, batch)
        return v[..., None].expand_as(vals), state


class StallSwitcher(Attack):
    """Batched ``attacks.StallSwitcher`` over trim-hugging up and down."""

    name = "switching"

    def __init__(self, a=0.99, window=50, tol=3e-3):
        self.a, self.window, self.tol = a, window, tol

    def init(self, batch):
        z = torch.zeros(batch.B, device=batch.device)
        hist = torch.zeros(self.window + 1, batch.B, device=batch.device, dtype=batch.dtype)
        return (torch.ones(batch.B, device=batch.device, dtype=torch.bool), z.long(), hist)

    def __call__(self, t, x, vals, batch, state):
        up, last, hist = state
        m = batch.mean(x)
        hist = torch.cat([hist[1:], m[None]], 0)       # hist[0] is the mean `window` steps ago
        if t >= self.window:
            flip = ((t - last) >= self.window) & ((m - hist[0]).abs() < self.tol)
            up = up ^ flip
            last = torch.where(flip, torch.full_like(last, t), last)
        a = torch.where(up, self.a, -self.a)[:, None].expand_as(x)
        v, _ = band(a, x, vals, batch)
        return v[..., None].expand_as(vals), (up, last, hist)


def stealthy(instances, fake):
    """[1, Thm. 4.4] in the batched simulator: the adversaries run the regular
    update exactly, with minimizer ``fake``. That is the same as making them
    regular nodes with a_j = fake and excluding them from the metrics."""
    out = []
    for I in instances:
        J = dict(I)
        ev = np.ones(I["adj"].shape[0], dtype=bool)
        ev[list(I["adv"])] = False
        J["a"] = np.array(I["a"], dtype=float).copy()
        J["a"][list(I["adv"])] = fake
        J["adv"], J["eval"] = [], ev
        out.append(J)
    return out


# ---------------------------------------------------------------- simulation

def power_alpha(c=0.5, p=0.6):
    return lambda t: c / (t + 1.0) ** p


def step(t, x, batch, rule, alpha, attack, astate):
    vals = batch.gather(x)
    if attack is not None:
        av, astate = attack(t, x, vals, batch, astate)
        vals = torch.where(batch.from_adv, av, vals)
    lie = getattr(attack, "grads", None)
    return torch.where(batch.reg, descend(rule, x, vals, batch, t, alpha(t), attack is not None, lie), x), astate


def gradient_lie(x, g, vals, gvals, batch):
    """Batched ``dynamics.gradient_lie``: per slot, the extreme honest gradient
    in the receiver's pool on the side that helps the adversary's push."""
    regslot = batch.nmask & ~batch.from_adv
    inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    gmin = torch.minimum(torch.where(regslot, gvals, inf).amin(-1), g)
    gmax = torch.maximum(torch.where(regslot, gvals, -inf).amax(-1), g)
    push = torch.sign(vals - x[..., None])
    return torch.where(push > 0, gmin[..., None], torch.where(push < 0, gmax[..., None], g[..., None]))


def descend(rule, x, vals, batch, t, at, attacked=True, lie=None):
    """New states of all nodes given the received values (regular rows are used).
    ``lie(x, g, vals, gvals, batch)``: fake gradients on adversary slots, for rules
    that exchange gradients (default ``gradient_lie``)."""
    if getattr(rule, "exchanges_gradients", False):
        g = batch.grad(x)                                   # each node's gradient at its own state
        gvals = batch.gather(g)
        if attacked:
            gvals = torch.where(batch.from_adv, (lie or gradient_lie)(x, g, vals, gvals, batch), gvals)
        return rule.update(x, vals, g, gvals, batch, t, at)
    y = rule.aggregate(x, vals, batch, t, at)
    return y - at * batch.grad(x if rule.grad_at_own else y)


def run(batch, rule, T, alpha, attack=None, burn=None, ckpt=0):
    """Simulate T steps; returns final states and running metrics.

    ``burn``: metrics on regular states (hull violation, range) start here
    (default T // 2). ``ckpt`` > 0 recomputes chunks of that many steps in
    the backward pass instead of storing them (for training through long runs).
    """
    burn = T // 2 if burn is None else burn
    rule.reset(batch)
    astate = attack.init(batch) if attack is not None else ()
    x = batch.x0.clone()
    means = [batch.mean(x)]
    big = torch.finfo(x.dtype).max
    lo = torch.full((batch.B,), big, device=x.device, dtype=x.dtype)
    hi = -lo
    if burn == 0:
        lo = torch.where(batch.ev, x, big).amin(1)
        hi = torch.where(batch.ev, x, -big).amax(1)

    def chunk(t0, t1, x, *astate):
        out = []
        for t in range(t0, t1):
            x, astate = step(t, x, batch, rule, alpha, attack, astate)
            out.append(x)
        return (torch.stack(out), *astate)

    t = 0
    while t < T:
        t1 = min(T, t + (ckpt or T))
        if ckpt and torch.is_grad_enabled():
            res = checkpoint(chunk, t, t1, x, *astate, use_reentrant=False)
        else:
            res = chunk(t, t1, x, *astate)
        xs, astate = res[0], tuple(res[1:])
        evf = batch.ev
        means.append((xs * evf).sum(-1) / evf.sum(-1))
        if t1 >= burn:                                      # xs[j] is the state at time t + 1 + j
            seg = xs[max(0, burn - t - 1):]
            lo = torch.minimum(lo, torch.where(evf, seg, big).amin((0, 2)))
            hi = torch.maximum(hi, torch.where(evf, seg, -big).amax((0, 2)))
        x, t = xs[-1], t1
    mean = torch.cat([means[0][None], *means[1:]], 0)
    xr = torch.where(batch.ev, x, torch.nan)
    return {
        "x": x,
        "mean": mean,                                        # (T+1, B)
        "error": (mean[-1] - batch.x_star).abs(),
        "spread": xr.nan_to_num(-big).amax(1) - xr.nan_to_num(big).amin(1),
        "min": lo, "max": hi,                                # regular states, t >= burn
        "hull_violation": torch.clamp(torch.maximum(batch.hull_lo - lo, hi - batch.hull_hi), min=0),
        "oscillation": (mean[burn + 1:] - mean[burn:-1]).abs().sum(0),
    }


def range_excess(batch, rule, attack=None, T=300):
    """Pure-consensus safety check of Phases 2-3: with alpha = 0 the regular
    states must stay inside the regular nodes' initial range."""
    r = run(batch, rule, T, lambda t: 0.0, attack, burn=0)
    big = torch.finfo(batch.dtype).max
    lo0 = torch.where(batch.ev, batch.x0, big).amin(1)
    hi0 = torch.where(batch.ev, batch.x0, -big).amax(1)
    return torch.clamp(torch.maximum(lo0 - r["min"], r["max"] - hi0), min=0)
