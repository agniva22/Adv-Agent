"""Figures 1-3 of the paper.

Stage 1 simulates and writes results/figures/data.npz; stage 2 draws each panel at its
printed size (4.2 cm x 2.7 cm, font sizes in pt, text outlined) as results/figures/<name>.{eps,png}:

    network               example network (n = 20, F = 1), Fig. 1(a)
    states_dgd            regular states under DGD, fixed adversary (captured)
    states_ours           regular states under memory + fallback, its best-response attacker
    mean_lf_ours          regular mean: LF vs memory + fallback
    error_F1, error_F2    error |x_bar(t) - x*| over time, 100 unseen graphs (Fig. 2)
    error_legend          shared legend of error_F1 / error_F2
    paired_F1, paired_F2  per-graph error, LF vs memory + fallback (Fig. 3)

The example network is the unseen F = 1 graph with the median improvement of
memory + fallback over LF (not cherry-picked). Stage 2 needs Ghostscript and pdftops.

    python experiments/11_figures.py [--plot-only]
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import torch

matplotlib.use("Agg")
matplotlib.rcParams["ps.fonttype"] = 42   # embed text as TrueType, not Type 3
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import networkx as nx

from advagent import checkpoints as C
from advagent import learn as Lr
from advagent import torchsim as ts

OUT = C.FIGURES
DATA = OUT / "data.npz"
EVAL = C.EVALUATION / "results.json"
BR = C.BR
T = 2000
# Fixed order: an entity keeps its colour in every figure. Ours first.
METHODS = ["memory + fallback", "memory + fallback (game)", "LF", "LF-Metropolis", "Trimmed mean", "Su-Vaidya"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STYLES = ["-", "-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]
SHORT = {"memory + fallback": "Ours", "memory + fallback (game)": "Ours (game)", "LF": "LF",
         "LF-Metropolis": "LF-Metropolis", "Trimmed mean": "Trimmed mean", "Su-Vaidya": "Su–Vaidya"}
# Labels of the separate legend file (error_legend) only.
LEGEND_LABELS = {**SHORT, "memory + fallback": "Ours (M+F)", "memory + fallback (game)": "Ours (M+F with game)",
                 "Su-Vaidya": "BR-MO"}
STATE_BLUE, HULL_FILL, ADV_RED, REG_FILL = "#2a78d6", "#cde2fb", "#d03b3b", "#86b6ef"

CM = 1 / 2.54
FIGSIZE = (4.2 * CM, 2.7 * CM)
LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, INSET_TICK_SIZE = 8, 7, 6.5, 5




def rule_of(name):
    return ts.BASELINES[name]() if name in ts.BASELINES else C.defender(name)


# ------------------------------------------------------------------ stage 1: simulation

@torch.no_grad()
def error_curves(name, F, insts):
    """Mean over graphs of |x_bar(t) - x*| under the method's best-response attacker,
    taking the worse push direction per graph (as in the evaluation)."""
    rule, pol = rule_of(name), C.best_response_attacker(name, F)
    curves, finals = [], []
    for i in range(0, len(insts), 10):
        batch = ts.Batch(insts[i:i + 10])
        runs = []
        for dr in (1, -1):
            torch.manual_seed(0)
            r = ts.run(batch, rule, T, ts.power_alpha(), Lr.LearnedAttack(pol, direc=dr))
            runs.append((r["mean"] - batch.x_star[None]).abs())          # (T+1, B)
        pick = runs[0][-1] >= runs[1][-1]
        e = torch.where(pick[None], runs[0], runs[1])
        curves.append(e.numpy())
        finals.append(e[-1].numpy())
    print(f"  curves: {name}, F={F}", flush=True)
    return np.concatenate(curves, 1).mean(1), np.concatenate(finals)


@torch.no_grad()
def trajectory(rule, inst, attack):
    """All states over time for one instance (B = 1)."""
    batch = ts.Batch([inst])
    rule.reset(batch)
    astate = attack.init(batch) if attack is not None else ()
    x, alpha, xs = batch.x0.clone(), ts.power_alpha(), [batch.x0[0].numpy().copy()]
    for t in range(T):
        x, astate = ts.step(t, x, batch, rule, alpha, attack, astate)
        xs.append(x[0].numpy().copy())
    return np.array(xs), float(batch.x_star[0])


def worse_trajectory(name, inst, F):
    pol = C.best_response_attacker(name, F)
    best = None
    for dr in (1, -1):
        torch.manual_seed(0)
        xs, xstar = trajectory(rule_of(name), inst, Lr.LearnedAttack(pol, direc=dr))
        reg = np.ones(xs.shape[1], bool)
        reg[list(inst["adv"])] = False
        err = abs(xs[-1, reg].mean() - xstar)
        if best is None or err > best[0]:
            best = (err, xs)
    return best[1]


def simulate():
    torch.set_num_threads(4)
    out = {}
    for F in (1, 2):
        insts = Lr.train_pool(F, 100, offset=300_000)   # the unseen graphs of 06_evaluate.py
        for name in METHODS:
            curve, finals = error_curves(name, F, insts)
            out[f"curve|{name}|F{F}"] = curve
            out[f"final|{name}|F{F}"] = finals
    # Example: the unseen F = 1 graph with the median improvement of memory + fallback over LF.
    grid = json.loads(EVAL.read_text())["grid"]
    gain = np.array(grid[f"LF|F1|{BR}"]["error"]) - np.array(grid[f"memory + fallback|F1|{BR}"]["error"])
    k = int(np.argsort(gain)[len(gain) // 2])
    inst = Lr.train_pool(1, 100, offset=300_000)[k]
    torch.manual_seed(0)
    out["ex|DGD"], xstar = trajectory(ts.BASELINES["DGD"](), inst, ts.Fixed(50.0))
    out["ex|LF"] = worse_trajectory("LF", inst, 1)
    out["ex|memory + fallback"] = worse_trajectory("memory + fallback", inst, 1)
    reg = np.ones(inst["adj"].shape[0], bool)
    reg[list(inst["adv"])] = False
    out.update({"ex_adj": inst["adj"], "ex_reg": reg, "ex_a": inst["a"], "ex_xstar": np.array(xstar), "ex_index": np.array(k)})
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA, **out)
    for F in (1, 2):   # consistency with Table II (same graphs, same attackers)
        print(f"  F={F} final BR errors:", {n: round(float(out[f'final|{n}|F{F}'].mean()), 3) for n in METHODS})



# ------------------------------------------------------------------ stage 2: panels

def legend_above(ax, ncol=None):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=ncol or len(labels),
              frameon=True, prop={"weight": "bold", "size": LEGEND_SIZE}, handlelength=1.4, handletextpad=0.3,
              columnspacing=0.6, borderpad=0.2, borderaxespad=0.2, framealpha=1.0)


def fit_to_paper(ax, insets=(), nbins=3):
    ax.xaxis.label.set_size(LABEL_SIZE)
    ax.yaxis.label.set_size(LABEL_SIZE)
    ax.xaxis.labelpad = 1
    ax.yaxis.labelpad = 1
    for a in (ax, *insets):
        main = a is ax
        a.tick_params(axis="both", which="both", width=0.5, length=2.0 if main else 1.2, pad=1.5)
        for spine in a.spines.values():
            spine.set_linewidth(0.6 if main else 0.5)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.tick_params(axis="x", which="major", pad=2.5)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=nbins, steps=[1, 2, 5, 10]))
    for axins in insets:
        # only the zoomed time window is labelled; y-values do not fit at this size
        axins.set_yticks([])
        t0, t1 = axins.get_xlim()
        axins.set_xticks([t0, t1])
        axins.set_xticklabels([f"{t0:g}", f"{t1:g}"])
        axins.tick_params(axis="x", labelsize=INSET_TICK_SIZE, pad=1.0, labeltop=True, labelbottom=False)
        axins.get_xticklabels()[0].set_ha("left")
        axins.get_xticklabels()[-1].set_ha("right")


def add_inset(ax, t, curves, t0, t1, bounds=(0.42, 0.42, 0.53, 0.38), pad=0.15):
    """Zoom on the time window [t0, t1]. ``curves`` is a list of (y, plot kwargs)."""
    axins = ax.inset_axes(bounds)
    sel = (t >= t0) & (t <= t1)
    lo = min(np.min(y[sel]) for y, _ in curves)
    hi = max(np.max(y[sel]) for y, _ in curves)
    for y, kw in curves:
        axins.plot(t[sel], y[sel], **{**kw, "lw": kw.get("lw", 0.8) * 0.6})
    m = pad * max(hi - lo, 1e-3)
    axins.set_xlim(t0, t1)
    axins.set_ylim(lo - m, hi + m)
    ax.indicate_inset_zoom(axins, edgecolor="0.4", lw=0.4)
    return axins


def save(fig, name):
    fig.set_layout_engine("constrained", w_pad=0.01, h_pad=0.01)
    eps = OUT / f"{name}.eps"
    fig.savefig(eps, format="eps")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)
    outline_text(eps)


def outline_text(eps: Path) -> None:
    """Convert all text in the EPS to vector outlines, so the paper PDF gets no embedded (Type 3) fonts."""
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "fig.pdf"
        subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-dNoOutputFonts", "-dEPSCrop",
                        "-sDEVICE=pdfwrite", f"-sOutputFile={pdf}", str(eps)], check=True)
        subprocess.run(["pdftops", "-eps", str(pdf), str(eps)], check=True)


def plot():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA)
    reg = d["ex_reg"]
    a = d["ex_a"]
    xstar = float(d["ex_xstar"])
    lo_h, hi_h = a[reg].min(), a[reg].max()
    T = d["ex|DGD"].shape[0] - 1
    t = np.arange(T + 1, dtype=float)

    # ---- network
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=180)
    G = nx.from_numpy_array(d["ex_adj"].astype(int))
    pos = nx.spring_layout(G, seed=3)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#b5b5b5", width=0.35)
    nx.draw_networkx_nodes(G, pos, nodelist=list(np.flatnonzero(reg)), node_color=REG_FILL, node_size=9,
                           edgecolors=STATE_BLUE, linewidths=0.4, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=list(np.flatnonzero(~reg)), node_color=ADV_RED, node_shape="s",
                           node_size=10, edgecolors=ADV_RED, linewidths=0.4, ax=ax)
    ax.plot([], [], "o", color=REG_FILL, mec=STATE_BLUE, ms=3, mew=0.4, label="regular")
    ax.plot([], [], "s", color=ADV_RED, ms=3, label="adversarial")
    ax.axis("off")
    legend_above(ax)
    save(fig, "network")

    # ---- regular states: DGD (captured) and ours (inside H)
    for key, name, inset in (("ex|DGD", "states_dgd", None), ("ex|memory + fallback", "states_ours", (500, 2000))):
        xs = d[key][:, reg]
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=180)
        ax.axhspan(lo_h, hi_h, color=HULL_FILL, lw=0, label=r"$\mathcal{H}$")
        for i in range(xs.shape[1]):
            ax.plot(t, xs[:, i], color=STATE_BLUE, lw=0.5, label=r"$x_i$" if i == 0 else None)
        ax.axhline(xstar, color="k", lw=0.5, ls="--", label=r"$x^\star$")
        ax.set_xlim(0, T)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x_i(t)$")
        insets = []
        if inset is None:
            ax.set_ylim(-8, 48)
        else:
            curves = [(xs[:, i], dict(color=STATE_BLUE, lw=0.8)) for i in range(xs.shape[1])]
            curves.append((np.full_like(t, xstar), dict(color="k", lw=0.8, ls="--")))
            axins = add_inset(ax, t, curves, *inset, bounds=(0.42, 0.06, 0.53, 0.30))
            axins.axhspan(lo_h, hi_h, color=HULL_FILL, lw=0, zorder=0)
            insets.append(axins)
        legend_above(ax)
        fit_to_paper(ax, insets)
        save(fig, name)

    # ---- regular mean: LF vs ours
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=180)
    curves = []
    for key, mname in (("ex|LF", "LF"), ("ex|memory + fallback", "memory + fallback")):
        k = METHODS.index(mname)
        y = d[key][:, reg].mean(1)
        kw = dict(color=COLORS[k], ls=STYLES[k], lw=0.8)
        ax.plot(t, y, label=SHORT[mname], **kw)
        curves.append((y, kw))
    ax.axhline(xstar, color="k", lw=0.5, ls="--", label=r"$x^\star$")
    ax.set_xlim(0, T)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\bar{x}_{\mathcal{R}}(t)$")
    axins = add_inset(ax, t, curves, 1000, 2000, bounds=(0.42, 0.06, 0.53, 0.30), pad=0.4)
    legend_above(ax)
    fit_to_paper(ax, [axins])
    save(fig, "mean_lf_ours")

    # ---- error over time, F = 1 and F = 2 (legend in a separate file)
    for F in (1, 2):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=180)
        curves = []
        for mname, col, ls in zip(METHODS, COLORS, STYLES):
            y = d[f"curve|{mname}|F{F}"]
            kw = dict(color=col, ls=ls, lw=1.0 if mname.startswith("memory") else 0.8)
            ax.plot(t, y, label=SHORT[mname], **kw)
            curves.append((y, kw))
        ax.set_xlim(0, T)
        ax.set_ylim(0, None)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$|\bar{x}_{\mathcal{R}}(t) - x^\star|$")
        axins = add_inset(ax, t, curves, 1000, 2000, bounds=(0.42, 0.06, 0.53, 0.32))
        fit_to_paper(ax, [axins])
        save(fig, f"error_F{F}")

    fig = plt.figure(figsize=(8.4 * CM, 0.75 * CM), dpi=180)
    handles = [plt.Line2D([], [], color=c, ls=s, lw=1.0 if m.startswith("memory") else 0.8)
               for m, c, s in zip(METHODS, COLORS, STYLES)]
    fig.legend(handles, [LEGEND_LABELS[m] for m in METHODS], loc="center", ncol=3, frameon=True,
               prop={"weight": "bold", "size": LEGEND_SIZE}, handlelength=1.8, handletextpad=0.3,
               columnspacing=0.6, borderpad=0.2, framealpha=1.0)
    save(fig, "error_legend")

    # ---- per-graph error, LF vs ours
    grid = json.loads(EVAL.read_text())["grid"]
    for F in (1, 2):
        x = np.array(grid[f"LF|F{F}|{BR}"]["error"])
        y = np.array(grid[f"memory + fallback|F{F}|{BR}"]["error"])
        m = max(x.max(), y.max()) * 1.05
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=180)
        ax.plot([0, m], [0, m], color="k", lw=0.5, ls="--", label="equal error")
        ax.scatter(x, y, s=3, color=STATE_BLUE, linewidths=0, zorder=3, label="graph")
        ax.text(0.04, 0.95, f"ours lower: {int((y < x).sum())}/100", transform=ax.transAxes, ha="left",
                va="top", fontsize=LEGEND_SIZE, fontweight="bold")
        ax.set_xlim(0, m)
        ax.set_ylim(0, m)
        ax.set_xlabel("LF error")
        ax.set_ylabel("Ours error")
        legend_above(ax)
        fit_to_paper(ax)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
        save(fig, f"paired_F{F}")

    print("saved to", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true", help="redraw from results/figures/data.npz")
    args = ap.parse_args()
    if not args.plot_only:
        simulate()
    plot()
