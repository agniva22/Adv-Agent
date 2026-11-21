"""Plots shared by several experiments."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import bench


def attacks_for(res, F):
    return [a for a in list(bench.SCRIPTED) + ["RL attacker (vs LF)"] if any(k[1] == F and k[2] == a for k in res)]


def heatmap(res, methods, path, title):
    names = list(methods)
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(names) + 2.6), sharey=True)
    norm = matplotlib.colors.LogNorm(1e-2, 2)
    for ax, F in zip(axes, (1, 2)):
        attacks = attacks_for(res, F)
        E = np.array([[res[(n, F, a)]["error"].mean() for a in attacks] for n in names])
        im = ax.imshow(np.clip(E, 1e-2, None), cmap="Blues", norm=norm, aspect="auto")
        for i, n in enumerate(names):
            for j, a in enumerate(attacks):
                f = bench.flag(res[(n, F, a)])
                ax.text(j, i, f"{E[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="#1f1f1f" if norm(max(E[i, j], 1e-2)) < 0.6 else "white")
                if f:
                    ax.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False,
                                               ec="#d6311f" if f == "unsafe" else "#1f1f1f", lw=1.6,
                                               ls="-" if f == "unsafe" else "--"))
        ax.set_xticks(range(len(attacks)), attacks, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_title(f"F = {F}", fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("mean error |x̄ − x*| (log)")
    handles = [plt.Rectangle((0, 0), 1, 1, fill=False, ec="#d6311f", lw=1.6, label="unsafe in some run"),
               plt.Rectangle((0, 0), 1, 1, fill=False, ec="#1f1f1f", lw=1.6, ls="--", label="no consensus in some run")]
    fig.legend(handles=handles, loc="upper left", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.02, -0.1))
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
