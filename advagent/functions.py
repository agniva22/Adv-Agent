"""Local objective functions f_i : R -> R, vectorised over nodes.

Each node holds either a capped quadratic (x - a_i)^2, whose gradient is
clipped to [-L, L] (the paper's footnote 4: capping does not move minimizers
of convex combinations when L is large), or c_i |x - a_i|.
"""

import numpy as np
from scipy.optimize import minimize_scalar

QUAD, ABS = 0, 1


class LocalFunctions:
    def __init__(self, a, kind=None, scale=None, L=1e3):
        self.a = np.asarray(a, dtype=float)
        n = self.a.size
        self.kind = np.full(n, QUAD) if kind is None else np.asarray(kind)
        self.scale = np.ones(n) if scale is None else np.asarray(scale, dtype=float)
        self.L = L

    @classmethod
    def quadratic(cls, a, L=1e3):
        return cls(a, L=L)

    def grad(self, x, idx=None):
        """A subgradient of f_i at x_i, for nodes ``idx`` (all nodes by default)."""
        a, kind, c = self._sel(idx)
        d = x - a
        g = np.where(kind == QUAD, 2 * d, c * np.sign(d))
        return np.clip(g, -self.L, self.L)

    def value(self, x, idx=None):
        a, kind, c = self._sel(idx)
        d = np.abs(x - a)
        # Capped quadratic: quadratic up to |d| = L/2, then linear with slope L.
        knee = self.L / 2
        quad = np.where(d <= knee, d**2, knee**2 + self.L * (d - knee))
        return np.where(kind == QUAD, quad, c * d)

    def _sel(self, idx):
        if idx is None:
            return self.a, self.kind, self.scale
        return self.a[idx], self.kind[idx], self.scale[idx]

    def minimizer_hull(self, idx):
        """[min, max] of the local minimizers of nodes ``idx`` (the set of [1, Thm. 7.1])."""
        return float(self.a[idx].min()), float(self.a[idx].max())

    def global_minimizer(self, idx):
        """Minimizer x* of the average of f_i over nodes ``idx``."""
        lo, hi = self.minimizer_hull(idx)
        if lo == hi:
            return lo
        res = minimize_scalar(
            lambda x: self.value(np.full(len(idx), x), idx).mean(),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-10},
        )
        return float(res.x)

    def average_value(self, x, idx):
        return float(self.value(np.full(len(idx), x), idx).mean())
