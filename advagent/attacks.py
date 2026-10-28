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
