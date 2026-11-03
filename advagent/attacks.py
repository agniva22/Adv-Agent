"""Scripted adversary behaviours. Each attack is a callable ``attack(t, x, ctx)``
returning the values the adversaries send (see ``dynamics.simulate``)."""

import numpy as np


class Fixed:
    """Every adversary broadcasts a constant (the stubborn agent of [1, Prop. 4.3])."""

    def __init__(self, value):
        self.value = value

    def __call__(self, t, x, ctx):
        return np.full(ctx["adv"].size, self.value, dtype=float)


class UniformRandom:
    """Malicious: one fresh uniform value per adversary per step.
    Byzantine: a fresh value per (receiver, adversary) pair."""

    def __init__(self, lo, hi, byzantine=False):
        self.lo, self.hi, self.byzantine = lo, hi, byzantine

    def __call__(self, t, x, ctx):
        n, k = x.size, ctx["adv"].size
        shape = (n, k) if self.byzantine else (k,)
        return ctx["rng"].uniform(self.lo, self.hi, size=shape)


class FarthestEvader:
    """Byzantine attack on rules that drop the F values farthest from the
    receiver's own value, such as DRC-RSSG [10].

    To each regular receiver i the adversary sends x_i + direction * frac * D_i,
    where D_i is the distance from x_i to its farthest regular in-neighbour.
    With frac < 1 the value is never among the farthest, so it is kept, yet for
    the receiver holding the extreme value it lies outside the regular nodes'
    range. LF drops it, because it is the most extreme value on its side of x_i.
    """

    def __init__(self, direction=1.0, frac=0.9):
        self.direction, self.frac = direction, frac

    def __call__(self, t, x, ctx):
        adj, reg = ctx["adj"], ctx["regular"]
        n, k = x.size, ctx["adv"].size
        D = np.array([np.abs(x[adj[i] & reg] - x[i]).max(initial=0.0) for i in range(n)])
        return np.broadcast_to((x + self.direction * self.frac * D)[:, None], (n, k))


class Switching:
    """The oscillation attack of [1, Ex. 7.2] on the 5-node complete graph.

    ``low`` are the regular nodes sharing one function (v1-v3), ``odd`` the
    node with the other function (v4). Two phases:

      * shadow: copy the value of ``low[0]``. With F = 1 the low nodes then
        drop v4 as the single extreme value, so everyone drifts to f_a's
        minimizer.
      * expose: send a value above v4. The low nodes now drop the adversary
        instead and listen to v4, so everyone drifts towards the minimizer
        of a mix of f_a and f_b.

    The omniscient adversary switches with hysteresis on the regular nodes'
    mean: shadow -> expose once it falls below ``lo``, expose -> shadow once it
    rises above ``hi``. ``lo``/``hi`` must lie strictly inside the two phase
    equilibria (about 0 and 2.45 for the functions of Ex. 7.2), so that each
    phase is always reached and the oscillation never dies out.
    """

    def __init__(self, low, odd, lo=0.25, hi=2.0, offset=1.0):
        self.low, self.odd = list(low), odd
        self.lo, self.hi, self.offset = lo, hi, offset
        self.phase, self.switches = "shadow", []

    def __call__(self, t, x, ctx):
        mean = x[ctx["regular"]].mean()
        if self.phase == "shadow" and mean < self.lo:
            self.phase = "expose"
            self.switches.append(t)
        elif self.phase == "expose" and mean > self.hi:
            self.phase = "shadow"
            self.switches.append(t)
        value = x[self.low[0]] if self.phase == "shadow" else x[self.odd] + self.offset
        return np.full(ctx["adv"].size, value)


class Stealthy:
    """The undetectable attack of [1, Thm. 4.4]: each adversary runs the
    defenders' own update rule exactly, as a regular node would, but descends a
    fake objective (x - fake)^2 instead of a real one.

    No algorithm can tell such a node from a regular one, so its fake
    minimizer is indistinguishable from a genuine local objective; whatever a
    method converges to, it can be pulled towards ``fake``. Malicious (one
    broadcast value per adversary). ``alpha`` must be the step-size schedule
    the regular nodes use.
    """

    def __init__(self, fake, alpha):
        self.fake, self.alpha = fake, alpha
        self.z = None

    def __call__(self, t, x, ctx):
        adv, rule = ctx["adv"], ctx["rule"]
        if self.z is None:
            self.z = x[adv].astype(float)
        sent = self.z.copy()
        # x[adv] holds what adversaries sent last step; they see each other's current state.
        x = x.copy()
        x[adv] = self.z
        L = ctx["funcs"].L
        if getattr(rule, "exchanges_gradients", False):
            # Honest gradients of the fake objective, at the state each adversary sends.
            g = ctx["grads"].copy()
            g[adv] = np.clip(2 * (self.z - self.fake), -L, L)
            self.gsent = g[adv]
            self.z = np.array([rule.update(a, z, x[ctx["nbrs"][a]], g[a], g[ctx["nbrs"][a]], self.alpha(t), ctx)
                               for a, z in zip(adv, self.z)])
            return sent
        y = np.array([rule.aggregate(a, z, x[ctx["nbrs"][a]], ctx) for a, z in zip(adv, self.z)])
        at = self.z if getattr(rule, "grad_at_own", False) else y
        self.z = y - self.alpha(t) * np.clip(2 * (at - self.fake), -L, L)
        return sent

    def grads(self, t, x, ctx):
        return self.gsent


class TrimHugger:
    """Byzantine: send each regular receiver i the most extreme value that the
    trimming rules still keep.

    Let r_F be the F-th most extreme regular in-neighbour value beyond x_i in
    ``direction``. The adversary sends x_i + frac * (r_F - x_i). With frac < 1
    at least F regular values lie further out, so LF, Su-Vaidya and DRC-RSSG
    all drop those instead and keep the adversary's value. If fewer than F
    regular values lie beyond x_i, any value beyond would be trimmed, so it
    sends x_i (no effect).
    """

    def __init__(self, F, direction=1.0, frac=0.99):
        self.F, self.direction, self.frac = F, direction, frac

    def __call__(self, t, x, ctx):
        adj, reg = ctx["adj"], ctx["regular"]
        n, k = x.size, ctx["adv"].size
        out = x.astype(float)
        for i in np.flatnonzero(reg):
            beyond = self.direction * (x[adj[i] & reg] - x[i])
            beyond = np.sort(beyond[beyond > 0])[::-1]
            if beyond.size >= self.F:
                out[i] = x[i] + self.direction * self.frac * beyond[self.F - 1]
        return np.broadcast_to(out[:, None], (n, k))


class StallSwitcher:
    """Graph-independent version of the Ex. 7.2 switching attack.

    Pushes the regular nodes one way with ``up`` until their mean stalls (moves
    less than ``tol`` over ``window`` steps), then switches to ``down``, and so
    on. Unlike ``Switching`` it needs no knowledge of the graph or of the
    phase equilibria. Records the switch times in ``switches``.
    """

    def __init__(self, up, down, window=50, tol=3e-3):
        self.up, self.down, self.window, self.tol = up, down, window, tol
        self.going_up, self.last, self.hist, self.switches = True, 0, [], []

    def __call__(self, t, x, ctx):
        m = x[ctx["regular"]].mean()
        self.hist.append(m)
        if t - self.last >= self.window and abs(m - self.hist[-1 - self.window]) < self.tol:
            self.going_up, self.last = not self.going_up, t
            self.switches.append(t)
        return (self.up if self.going_up else self.down)(t, x, ctx)
