"""Learned attacker (Formulation 5A) and shielded defender (Formulation 5B).

Attacker: one policy, shared by every (adversaries -> regular receiver) pair,
picks a band action a (see ``torchsim.band``) per receiver every ``k`` steps.
It sees the receiver's local view, its own previous action (a one-step memory,
enough to implement switching) and the omniscient global quantities of [1]'s
threat model (x*, the regular mean). It is trained with PPO for one of two
objectives (proposal Sec. 5A): ``bias`` pushes the regular mean in a direction
``dir`` drawn per episode; ``oscillation`` maximises its total variation over
the second half of the run.

Defender: one policy, shared by every regular node, scores the node itself
and each value LF retains. Scores become weights through the floored softmax
of the proposal (Sec. 5B), so every weight is at least c / (d_i + 1) and the
weights sum to one; ``docs/shield_theorem.md`` shows that this keeps the
guarantees of [1]. The simulator is differentiable in those weights, so the
defender is trained by backpropagation through the whole run.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from . import bench
from . import graphs as g
from . import torchsim as ts
from .torchsim import S


# ----------------------------------------------------------------- instances

def instance(seed, F, n=20):
    """The instance distribution of Phases 2-3: a random (2F+1)-robust graph,
    up to 3F F-local adversaries, minimizers and initial states in [-5, 5]."""
    rng = np.random.default_rng(seed)
    adj = g.random_robust(n, 2 * F + 1, rng, extra=1)
    adv = g.random_f_local_set(adj, F, 3 * F, rng)
    return {"adj": adj, "adv": adv, "a": rng.uniform(-5, 5, n), "x0": rng.uniform(-5, 5, n), "F": F}


def eval_instances(F, seeds=10):
    """Exactly the Phase 3 benchmark instances (``experiments/phase3_attacks.py``)."""
    return [instance(1000 * F + s, F) for s in range(seeds)]


def train_pool(F, size, offset=100_000):
    """Training instances; seeds are disjoint from the evaluation seeds."""
    return [instance(offset + 1000 * F + s, F) for s in range(size)]


def mlp(i, h, o, depth=2):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ attacker

class AttackerPolicy(nn.Module):
    n_local, n_global = 14, 5

    def __init__(self, hidden=64, amax=4.0):
        super().__init__()
        self.mu = mlp(self.n_local, hidden, 1)
        self.log_std = nn.Parameter(torch.tensor(-0.5))
        self.critic = mlp(self.n_global + self.n_local, hidden, 1)
        self.amax = amax

    def features(self, t, T, at, x, vals, batch, direc, prev_mean, a_prev):
        _, (hi, lo, D) = ts.band(torch.zeros_like(x), x, vals, batch)
        own = x[..., None]
        regslot = batch.nmask & ~batch.from_adv
        deg = batch.deg.clamp(min=1)
        m = batch.mean(x)
        dv = direc[:, None]
        grad = batch.grad(x)
        loc = torch.stack([
            torch.full_like(x, t / T), torch.full_like(x, math.log(max(at, 1e-8)) / 5), dv.expand_as(x),
            dv * (x - batch.x_star[:, None]) / S, (dv * (m - batch.x_star)[:, None] / S).expand_as(x),
            D / S, (hi - x) / S, (x - lo) / S,
            (regslot & (vals > own)).sum(-1) / deg, (regslot & (vals < own)).sum(-1) / deg,
            batch.n_adv_in / deg, (dv * (m - prev_mean)[:, None] / S).expand_as(x), dv * grad / (2 * S),
            dv * a_prev / self.amax,
        ], -1)
        xr = torch.where(batch.ev, x, torch.nan)
        spread = xr.nan_to_num(-1e9).amax(1) - xr.nan_to_num(1e9).amin(1)
        glob = torch.stack([torch.full_like(m, t / T), torch.full_like(m, math.log(max(at, 1e-8)) / 5),
                            direc * (m - batch.x_star) / S, spread / S, direc * (m - prev_mean) / S], -1)
        return loc, glob

    def acting(self, batch):
        """Receivers the adversaries can reach: (B, n) mask."""
        return batch.reg & (batch.n_adv_in > 0)

    def value(self, loc, glob, act):
        w = act[..., None].to(loc.dtype)
        pooled = (loc * w).sum(1) / w.sum(1).clamp(min=1)
        return self.critic(torch.cat([glob, pooled], -1))[..., 0]

    def dist(self, loc):
        return torch.distributions.Normal(self.mu(loc)[..., 0], self.log_std.exp())


class MemoryAwareAttacker(AttackerPolicy):
    """Attacker that also sees, for every receiver, what that receiver's
    ``TrustDefender`` memory says about the adversary slots (mean over them of
    the three memory channels; the side channel relative to ``dir``). The
    defender being attacked is set with ``watch``; against a rule without a
    memory the extra features are zero."""

    n_local = AttackerPolicy.n_local + 3

    def watch(self, rule):
        object.__setattr__(self, "rule", rule)   # a reference, not a submodule
        return self

    def features(self, t, T, at, x, vals, batch, direc, prev_mean, a_prev):
        loc, glob = super().features(t, T, at, x, vals, batch, direc, prev_mean, a_prev)
        mem = getattr(getattr(self, "rule", None), "mem", {}).get(t - 1)
        if mem is not None:
            mem = mem[..., :3]   # the shortest-memory channels
        if mem is None:
            extra = torch.zeros(*x.shape, 3, device=x.device, dtype=x.dtype)
        else:
            w = batch.from_adv[..., None].to(x.dtype)
            extra = (mem * w).sum(-2) / w.sum(-2).clamp(min=1)
            extra = torch.cat([extra[..., :2], extra[..., 2:] * direc[:, None, None]], -1)
        return torch.cat([loc, extra], -1), glob

    @classmethod
    def from_policy(cls, pol):
        """Same behaviour as ``pol`` (zero weights on the memory features)."""
        new = cls()
        state = new.state_dict()
        for k, v in pol.state_dict().items():
            if state[k].shape == v.shape:
                state[k] = v.clone()
            else:
                state[k] = torch.zeros_like(state[k])
                state[k][:, : v.shape[1]] = v
        new.load_state_dict(state)
        return new


class LearnedAttack(ts.Attack):
    """Run an ``AttackerPolicy`` as an attack (for evaluation or as a fixed
    opponent). Deterministic (mean action) unless ``sample``."""

    def __init__(self, policy, T=2000, k=10, direc=1.0, sample=False, name="RL attacker"):
        self.policy, self.T, self.k, self.direc, self.sample, self.name = policy, T, k, direc, sample, name
        self.alpha = ts.power_alpha()

    def init(self, batch):
        self.policy.to(batch.device)
        return (torch.zeros(batch.B, batch.n, device=batch.device, dtype=batch.dtype), batch.mean(batch.x0))

    def __call__(self, t, x, vals, batch, state):
        a, prev = state
        if t % self.k == 0:
            direc = torch.full((batch.B,), float(self.direc), device=batch.device, dtype=batch.dtype)
            with torch.no_grad():
                loc, _ = self.policy.features(t, self.T, self.alpha(t), x, vals, batch, direc, prev, a)
                dist = self.policy.dist(loc)
                a = (dist.sample() if self.sample else dist.mean).clamp(-self.policy.amax, self.policy.amax)
            prev = batch.mean(x)
        v, _ = ts.band(a, x, vals, batch)
        return v[..., None].expand_as(vals), (a, prev)


@torch.no_grad()
def attacker_rollout(policy, batch, rule, T, alpha, k, direc, objective="bias"):
    """One PPO rollout. Returns per-decision tensors (decisions first).

    Rewards per decision: ``bias``: dir * change of the regular mean (sums to
    dir * (x_bar(T) - x_bar(0))); ``oscillation``: |change of the mean over the
    decision's k steps|, from T/4 on. Measuring the change over a whole
    decision (not step by step) rewards sustained swings rather than the
    jitter that exploration noise alone produces."""
    rule.reset(batch)
    x = batch.x0.clone()
    act = policy.acting(batch)
    prev = batch.mean(x)
    buf = {"loc": [], "glob": [], "a": [], "logp": [], "v": [], "r": []}
    a = torch.zeros_like(x)
    osc = torch.zeros_like(direc)
    for t in range(T):
        vals = batch.gather(x)
        if t % k == 0:
            loc, glob = policy.features(t, T, alpha(t), x, vals, batch, direc, prev, a)
            dist = policy.dist(loc)
            a = dist.sample().clamp(-policy.amax, policy.amax)
            buf["loc"].append(loc), buf["glob"].append(glob), buf["a"].append(a)
            buf["logp"].append((dist.log_prob(a) * act).sum(1))
            buf["v"].append(policy.value(loc, glob, act))
            prev, m_start = batch.mean(x), batch.mean(x)
        av, _ = ts.band(a, x, vals, batch)
        vals = torch.where(batch.from_adv, av[..., None], vals)
        m_old = batch.mean(x)
        x = torch.where(batch.reg, ts.descend(rule, x, vals, batch, t, alpha(t)), x)
        if t >= T // 2:
            osc = osc + (batch.mean(x) - m_old).abs()
        if (t + 1) % k == 0 or t + 1 == T:
            if objective == "bias":
                buf["r"].append(direc * (batch.mean(x) - m_start))
            else:
                buf["r"].append((batch.mean(x) - m_start).abs() * float(t + 1 - k >= T // 4))
    out = {key: torch.stack(v) for key, v in buf.items()}
    out["act"] = act
    out["final"] = direc * (batch.mean(x) - batch.x_star) if objective == "bias" else osc
    return out


def train_attacker(rule_factory, pool, T=2000, k=10, iters=150, B=48, device="cpu", seed=0,
                   policy=None, lr=3e-4, log=None, epochs=4, minibatch=1024, objective="bias"):
    """PPO against a fixed defence. ``pool``: training instances (one F).
    Returns the trained policy and the learning curve: per iteration, the
    mean signed push dir * (x_bar(T) - x*) (bias) or the oscillation."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = policy or AttackerPolicy()
    policy.to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    alpha = ts.power_alpha()
    curve = []
    for it in range(iters):
        idx = rng.choice(len(pool), B, replace=len(pool) < B)
        batch = ts.Batch([pool[i] for i in idx], device=device)
        direc = torch.as_tensor(rng.choice([-1.0, 1.0], B), device=device, dtype=batch.dtype)
        ro = attacker_rollout(policy, batch, rule_factory(), T, alpha, k, direc, objective)
        # GAE with gamma = 1 (the return telescopes to the final push).
        r, v = ro["r"], ro["v"]
        Kd = r.shape[0]
        adv = torch.zeros_like(r)
        last = torch.zeros_like(r[0])
        for j in reversed(range(Kd)):
            nv = v[j + 1] if j + 1 < Kd else torch.zeros_like(v[0])
            delta = r[j] + nv - v[j]
            last = delta + 0.95 * last
            adv[j] = last
        ret = adv + v
        advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        loc, glob, a, logp0 = ro["loc"].flatten(0, 1), ro["glob"].flatten(0, 1), ro["a"].flatten(0, 1), ro["logp"].flatten()
        act = ro["act"].repeat(Kd, 1)
        A_, R_ = advn.flatten(), ret.flatten()
        N = A_.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(N, device=device)
            for s in range(0, N, minibatch):
                j = perm[s: s + minibatch]
                dist = policy.dist(loc[j])
                logp = (dist.log_prob(a[j]) * act[j]).sum(1)
                ratio = (logp - logp0[j]).exp()
                pl = -torch.minimum(ratio * A_[j], ratio.clamp(0.8, 1.2) * A_[j]).mean()
                vl = (policy.value(loc[j], glob[j], act[j]) - R_[j]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pl + 0.5 * vl - 1e-3 * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
        curve.append(float(ro["final"].mean()))
        if log and (it % 10 == 0 or it == iters - 1):
            log(f"  attacker it {it:4d}  {objective} {curve[-1]:+.3f}  std {policy.log_std.exp().item():.3f}")
    return policy.cpu(), curve


# ------------------------------------------------------------------ defender

class ShieldedDefender(nn.Module, ts.Rule):
    """Learned aggregation inside the LF shield (``shield=True``) or, for the
    ablation, a raw softmax over itself and every in-neighbour (``shield=False``)."""

    n_feat = 15

    def __init__(self, hidden=32, c=0.25, shield=True, name=None):
        nn.Module.__init__(self)
        self.phi = mlp(self.n_feat, hidden, hidden, depth=1)
        self.rho = mlp(2 * hidden, hidden, 1, depth=1)
        nn.init.zeros_(self.rho[-1].weight)
        nn.init.zeros_(self.rho[-1].bias)   # equal scores at init: exactly LF (or plain averaging)
        self.c, self.shield = c, shield
        self.name = name or ("Shielded RL" if shield else "Unshielded RL")

    def memory(self, x, vals, batch, t, keep):
        """Extra per-slot features (B, n, d+1, k), self first; none here."""
        return None

    def weights(self, x, vals, batch, at, t=0):
        own = x[..., None]
        keep = ts.lf_keep(own, vals, batch.nmask, batch.F) if self.shield else batch.nmask
        rel = torch.where(keep, vals - own, 0)
        scale = rel.abs().amax(-1, keepdim=True).clamp(min=1e-3)
        g = batch.grad(x)[..., None]
        ldi = torch.log1p(batch.deg)[..., None] / 3
        nk = keep.sum(-1, keepdim=True) / (batch.deg[..., None] + 1)
        la = math.log(max(at, 1e-8)) / 5
        # What the filter removed: the scores may use every received value
        # (the shield only restricts where the weight goes).
        raw = torch.where(batch.nmask, vals - own, 0)
        deg1 = batch.deg[..., None].clamp(min=1)
        up = (batch.nmask & (vals > own)).sum(-1, keepdim=True) / deg1
        dn = (batch.nmask & (vals < own)).sum(-1, keepdim=True) / deg1
        top, bot = raw.amax(-1, keepdim=True).clamp(min=0), raw.amin(-1, keepdim=True).clamp(max=0)
        ktop, kbot = rel.amax(-1, keepdim=True).clamp(min=0), rel.amin(-1, keepdim=True).clamp(max=0)
        node = torch.cat([g / (2 * S), torch.full_like(own, la), nk, up, dn,
                          torch.tanh(top / scale), torch.tanh(-bot / scale),
                          torch.tanh((top - ktop) / scale), torch.tanh((kbot - bot) / scale)], -1)   # (B, n, 9)
        edge = torch.cat([torch.stack([rel / scale, rel / S, torch.log1p(batch.nbr_deg) / 3, ldi.expand_as(rel),
                                       -torch.sign(g) * rel / scale, torch.zeros_like(rel)], -1),
                          node[..., None, :].expand(*rel.shape, node.shape[-1])], -1)
        selff = torch.cat([torch.zeros_like(own), torch.zeros_like(own), ldi, ldi, torch.zeros_like(own),
                           torch.ones_like(own), node], -1)[..., None, :]
        tok = torch.cat([selff, edge], -2)                       # (B, n, d+1, f), self first
        extra = self.memory(x, vals, batch, t, keep)
        if extra is not None:
            tok = torch.cat([tok, extra], -1)
        mask = torch.cat([torch.ones_like(keep[..., :1]), keep], -1)
        h = self.phi(tok)
        mf = mask[..., None].to(h.dtype)
        pooled = (h * mf).sum(-2, keepdim=True) / mf.sum(-2, keepdim=True)
        logit = torch.nan_to_num(self.rho(torch.cat([h, pooled.expand_as(h)], -1))[..., 0])  # finite scores (shield proof)
        sm = torch.softmax(logit.masked_fill(~mask, -torch.inf), -1)
        if self.shield:
            eta = self.c / (batch.deg[..., None] + 1)
            w = eta + (1 - mask.sum(-1, keepdim=True) * eta) * sm
            w = torch.where(mask, w, 0)
        else:
            w = sm
        if getattr(self, "gate", None) is not None:
            # Fallback: blend with LF's uniform weights. Both are valid shield
            # weights (>= eta_i, sum 1), so any convex combination is too.
            g = torch.sigmoid(torch.nan_to_num(self.gate(pooled[..., 0, :])))   # (B, n, 1); finite, so g in [0, 1]
            lf = mask / mask.sum(-1, keepdim=True)
            w = g * w + (1 - g) * lf
        return w, mask

    def aggregate(self, x, vals, batch, t, at):
        w, _ = self.weights(x, vals, batch, at, t)
        return w[..., 0] * x + (w[..., 1:] * vals).sum(-1)


class TrustDefender(ShieldedDefender):
    """Shielded defender with a per-neighbour memory and an LF fallback gate.

    For every in-neighbour slot the node keeps exponential averages (one set
    per factor in ``betas``) of how far that neighbour is from it relative to the others,
    how often LF trimmed it, and on which side it was. The memory is a
    function of past states only, stored per step so that gradient
    checkpointing replays it exactly. The shield proof allows scores that
    depend on history, so the guarantees are unchanged.
    """

    n_feat = ShieldedDefender.n_feat + 3

    def __init__(self, hidden=32, c=0.05, betas=(0.95,), name="Trust RL"):
        self.betas = tuple(betas)
        self.n_feat = ShieldedDefender.n_feat + 3 * len(self.betas)
        super().__init__(hidden, c, shield=True, name=name)
        self.gate = mlp(hidden, hidden, 1, depth=1)
        self.mem = {}

    def reset(self, batch):
        self.mem = {}

    def memory(self, x, vals, batch, t, keep):
        with torch.no_grad():
            diff = torch.where(batch.nmask, vals - x[..., None], 0)
            typical = diff.abs().sum(-1, keepdim=True) / batch.deg[..., None].clamp(min=1) + 1e-6
            cur = torch.stack([torch.tanh(diff.abs() / typical / 2), (batch.nmask & ~keep).to(x.dtype),
                               torch.sign(diff)], -1)
            prev = self.mem.get(t - 1)
            m = torch.cat([cur * (1 - b) if prev is None else b * prev[..., 3 * j: 3 * j + 3] + (1 - b) * cur
                           for j, b in enumerate(self.betas)], -1)
            self.mem[t] = m
        return torch.cat([torch.zeros_like(m[..., :1, :]), m], -2)


def defender_loss(r, batch):
    """Proposal 5B: squared error of every regular node at T, averaged (per episode)."""
    e = torch.where(batch.ev, r["x"] - batch.x_star[:, None], 0).pow(2)
    return e.sum(1) / batch.ev.sum(1)


def _window_losses(batch, rule, T, alpha, att, window, grad, ckpt=50):
    """Yield the per-node squared error at the end of every ``window`` steps.

    With ``grad``, each yielded loss is attached to its own window only (the
    state is detached between windows: truncated BPTT), and the caller must
    call backward before asking for the next window. Inside a window, chunks
    of ``ckpt`` steps are recomputed in the backward pass to save memory.
    """
    rule.reset(batch)
    astate = att.init(batch) if att is not None else ()
    x = batch.x0.clone()

    def chunk(t0, t1, x, *astate):
        for t in range(t0, t1):
            x, astate = ts.step(t, x, batch, rule, alpha, att, astate)
        return (x, *astate)

    for t0 in range(0, T, window):
        with torch.set_grad_enabled(grad):
            for c0 in range(t0, min(T, t0 + window), ckpt):
                c1 = min(T, t0 + window, c0 + ckpt)
                res = checkpoint(chunk, c0, c1, x, *astate, use_reentrant=False) if grad else chunk(c0, c1, x, *astate)
                x, astate = res[0], tuple(res[1:])
            yield defender_loss({"x": x}, batch)
        x, astate = x.detach(), tuple(s.detach() for s in astate)


def train_defender(defender, pools, attacks, T=2000, iters=300, B=24, device="cpu", seed=0, lr=3e-3,
                   ckpt=50, log=None, eval_fn=None, eval_every=50, loss_type="ratio", window=None):
    """Backpropagation through the simulator.

    ``pools``: {F: instances}. ``attacks``: list of (weight, factory) where
    factory(F) returns an attack spec as in ``bench.SCRIPTED``, or the string
    'stealthy' (fake minimizer drawn from [-50, 50]). Each iteration draws one
    F and one attack for the whole batch.

    ``loss_type``: 'ratio' (Phase 5: mean over episodes of loss / LF loss) or
    'diff' (batch loss minus LF's, divided by LF's batch loss: no per-episode
    division, so episodes where LF is nearly exact cannot dominate) or
    'constrained' ('diff' plus lambda_a * max(0, diff) for the drawn attack a,
    with lambda_a raised by dual ascent while the defender is worse than LF on
    a: "no worse than LF on any attack" as a soft constraint).
    ``window``: None for full backpropagation through T steps; otherwise
    truncated BPTT with a loss at the end of every window, later windows
    weighted more (weight k / K for window k of K).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    defender.to(device)
    opt = torch.optim.Adam(defender.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    alpha = ts.power_alpha()
    p = np.array([w for w, _ in attacks], dtype=float)
    p /= p.sum()
    Fs = list(pools)
    hist = []
    lam = {}
    for it in range(iters):
        F = Fs[rng.integers(len(Fs))]
        insts = [pools[F][i] for i in rng.choice(len(pools[F]), B, replace=False)]
        ai = rng.choice(len(attacks), p=p)
        spec = attacks[ai][1](F)
        if isinstance(spec, str) and spec == "stealthy":
            spec = ("stealthy", float(rng.uniform(-50, 50)))
        insts, att = bench.prepare(insts, spec)
        batch = ts.Batch(insts, device=device)
        opt.zero_grad()
        shown = None
        if window:
            torch.manual_seed(it)
            refs = list(_window_losses(batch, ts.LF(), T, alpha, att, window, grad=False))
            torch.manual_seed(it)
            K = len(refs)
            total = 0.0
            for k, (l, ref) in enumerate(zip(_window_losses(batch, defender, T, alpha, att, window, grad=True), refs)):
                part = (k + 1) / K * (l.mean() - ref.mean()) / (ref.mean() + 1e-4)
                part.backward()
                total += float(part.detach())
            loss = torch.tensor(total / sum((k + 1) / K for k in range(K)))
        else:
            # Loss relative to plain LF on the same episodes: batches differ a lot in
            # difficulty, and this keeps the gradient scale comparable across them.
            with torch.no_grad():
                torch.manual_seed(it)
                ref = defender_loss(ts.run(batch, ts.LF(), T, alpha, att), batch)
            torch.manual_seed(it)
            r = ts.run(batch, defender, T, alpha, att, ckpt=ckpt)
            own = defender_loss(r, batch)
            if loss_type == "ratio":
                loss = (own / (ref + 1e-4)).mean()
            else:
                loss = (own.mean() - ref.mean()) / (ref.mean() + 1e-4)
                shown = loss.detach()   # logged without the penalty
                if loss_type == "constrained":
                    la = lam.get((F, ai), 1.0)
                    lam[(F, ai)] = min(20.0, max(0.0, la + float(loss.detach())))
                    loss = loss + la * torch.relu(loss)
            loss.backward()
        nn.utils.clip_grad_norm_(defender.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(float(loss.detach() if shown is None else shown))
        if log and (it % 10 == 0 or it == iters - 1):
            log(f"  defender it {it:4d}  loss / LF loss {np.mean(hist[-10:]):.4f}")
        if eval_fn and (it + 1) % eval_every == 0:
            eval_fn(it + 1)
    return defender.cpu(), hist
