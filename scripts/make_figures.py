#!/usr/bin/env python
"""Generate every data figure of the paper from results/ (never edit the PDFs by hand).

python scripts/make_figures.py          # -> paper/figures/fig_*.pdf (+ .png previews)

Only finished runs (final.json present) are plotted; curves average over the finished seeds
(band = +-1 std when >1 seed). Colours follow the method, never its rank, and are fixed across
figures; methods outside the highlighted set are drawn in grey as context.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tables import PE_INFO  # noqa: E402
from stats import load_group  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "figures")

# fixed categorical order (validated reference palette, light mode); ours always slot 1
COLOR = {"viper": "#2a78d6", "rope_mixed": "#eda100", "string_cayley": "#1baf7a", "lepe": "#6250d6",
         "lookhere": "#e34948"}
MARKER = {"viper": "o", "rope_mixed": "s", "string_cayley": "D", "lepe": "^", "lookhere": "v"}
GREY = "#9a9893"
INK, INK2 = "#0b0b0b", "#52514e"
HIGHLIGHT = ["viper", "string_cayley", "lepe", "lookhere", "rope_mixed"]   # + ViPER w/o gates: 6 legend entries

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "grid.color": "#e4e3df", "grid.linewidth": 0.5, "lines.linewidth": 1.4, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL_W, TEXT_W = 3.25, 6.875          # CVPR column / text width in inches


def name(pe):
    return PE_INFO.get(pe, (pe, "", None))[0].replace("\\textbf{", "").replace("}", "").replace("$^\\ddagger$", "")


def style(pe):
    if pe in COLOR:
        return dict(color=COLOR[pe], lw=2.0 if pe == "viper" else 1.3, zorder=5 if pe == "viper" else 3)
    return dict(color=GREY, lw=0.8, zorder=1, alpha=0.8)


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, stem + ".pdf"))
    fig.savefig(os.path.join(OUT, stem + ".png"), dpi=200)
    plt.close(fig)


def finished(group):
    return {m: r for m, r in load_group(os.path.join(RES, group)).items() if m in PE_INFO}


def histories(group, method, seeds):
    hs = []
    for s in seeds:
        p = os.path.join(RES, group, f"{method}_s{s}", "history.json")
        if os.path.exists(p):
            hs.append([h["val_top1"] for h in json.load(open(p))])
    n = min(map(len, hs)) if hs else 0
    return np.array([h[:n] for h in hs])


# --------------------------------------------------------------------------- figures
def fig_curves(group="in100_deit_small"):
    runs = finished(group)
    if not runs:
        return "no runs"
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.1), gridspec_kw=dict(width_ratios=[1.35, 1]))
    order = [m for m in runs if m not in COLOR] + [m for m in HIGHLIGHT[::-1] if m in runs]
    for ax, lo in zip(axes, (1, 70)):
        for m in order:
            h = histories(group, m, runs[m].keys())
            if h.size == 0:
                continue
            ep = np.arange(1, h.shape[1] + 1)
            mu, sd = h.mean(0), h.std(0)
            st = style(m)
            sel = ep >= lo
            ax.plot(ep[sel], mu[sel], **st, label=name(m) if m in COLOR else None)
            if h.shape[0] > 1 and m in COLOR:
                ax.fill_between(ep[sel], (mu - sd)[sel], (mu + sd)[sel], color=st["color"], alpha=0.15, lw=0)
        ax.grid(True, axis="y")
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("val top-1 (%), 5% train holdout")
    axes[0].set_title("(a) full training", loc="left")
    axes[1].set_title("(b) last 30 epochs", loc="left")
    # direct labels at the right end of panel (b), legend below for the full set
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, -0.04))
    axes[0].text(0.99, 0.04, "grey: other baselines", transform=axes[0].transAxes, ha="right", color=INK2,
                 fontsize=6.5)
    save(fig, "fig_curves")
    return f"{len(runs)} methods"


def fig_curves_main(group="in100_deit_small", lo=70):
    """Column-width version for the main paper: the last 30 epochs only, ViPER w/o gates dashed."""
    runs = finished(group)
    if not runs:
        return "no runs"
    allr = load_group(os.path.join(RES, group))
    if "abl_no_gates" in allr:
        runs["abl_no_gates"] = allr["abl_no_gates"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    order = [m for m in runs if m not in COLOR and m != "abl_no_gates"] + ["abl_no_gates"] + \
        [m for m in HIGHLIGHT[::-1] if m in runs]
    for m in order:
        if m not in runs:
            continue
        h = histories(group, m, runs[m].keys())
        if h.size == 0:
            continue
        ep = np.arange(1, h.shape[1] + 1)
        mu, sd = h.mean(0), h.std(0)
        sel = ep >= lo
        if m == "abl_no_gates":
            ax.plot(ep[sel], mu[sel], color=COLOR["viper"], ls="--", lw=1.0, label="ViPER w/o gates", zorder=4)
            continue
        st = style(m)
        ax.plot(ep[sel], mu[sel], **st, label=name(m) if m in COLOR else None)
        if h.shape[0] > 1 and m in COLOR:
            ax.fill_between(ep[sel], (mu - sd)[sel], (mu + sd)[sel], color=st["color"], alpha=0.15, lw=0)
    ax.grid(True, axis="y")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val top-1 (%)")
    ax.set_ylim(73, 82.5)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.45, -0.24), fontsize=6,
              columnspacing=0.9, handlelength=2.0)
    save(fig, "fig_curves_main")
    return f"{len(runs)} methods, epochs {lo}-100"


def fig_resolution(group="in100_deit_small"):
    runs = finished(group)
    if not runs:
        return "no runs"
    res = sorted({int(r) for v in runs.values() for f in v.values() for r in f.get("multires", {})})
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.2))
    order = [m for m in runs if m not in COLOR] + [m for m in HIGHLIGHT[::-1] if m in runs]
    for m in order:
        ys = np.array([[f["multires"][str(r)]["extrapolate"]["top1"] for r in res] for f in runs[m].values()])
        mu = ys.mean(0)
        i224 = res.index(224)
        st = style(m)
        mk = dict(marker=MARKER.get(m, None), ms=3.5, mec="white", mew=0.5) if m in COLOR else {}
        axes[0].plot(res, mu, **st, **mk, label=name(m) if m in COLOR else None)
        axes[1].plot(res[i224:], (mu - mu[i224])[i224:], **st, **mk)
    for ax, rr in zip(axes, (res, [r for r in res if r >= 224])):
        ax.axvline(224, color=INK2, lw=0.6, ls=":")
        ax.set_xticks(rr)
        ax.set_xticklabels([str(r) for r in rr], rotation=0)
        ax.grid(True, axis="y")
        ax.set_xlabel("test resolution (px), trained at 224")
    axes[0].set_ylabel("test top-1 (%)")
    axes[1].set_ylabel("change vs. 224 (points)")
    axes[1].axhline(0, color=INK2, lw=0.6)
    axes[0].set_title("(a) accuracy", loc="left")
    axes[1].set_title("(b) change when testing above the training resolution", loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, -0.04))
    save(fig, "fig_supp_resolution")
    return f"{len(runs)} methods"


def fig_resolution_main(group="in100_deit_small", rmin=128, rmax=448):
    """Main-paper version: test top-1 from rmin to 448 (no relative panel), so the differences between the
    strong encodings are visible; the full range and the change w.r.t. 224 are in fig_supp_resolution."""
    runs = finished(group)
    if not runs:
        return "no runs"
    main_runs = _main_runs()
    if "abl_no_gates" in main_runs:
        runs = dict(runs, abl_no_gates=main_runs["abl_no_gates"])
    res = [r for r in sorted({int(r) for v in runs.values() for f in v.values() for r in f.get("multires", {})}) if rmin <= r <= rmax]
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    order = [m for m in runs if m not in COLOR and m != "abl_no_gates"] + ["abl_no_gates"] + [m for m in HIGHLIGHT[::-1] if m in runs]
    for m in order:
        if m not in runs:
            continue
        mu = np.array([[f["multires"][str(r)]["extrapolate"]["top1"] for r in res] for f in runs[m].values()]).mean(0)
        if m == "abl_no_gates":
            ax.plot(res, mu, color=COLOR["viper"], ls="--", lw=1.0, marker="o", ms=3, mfc="white", mec=COLOR["viper"],
                    label="ViPER w/o gates", zorder=4)
            continue
        st = style(m)
        mk = dict(marker=MARKER.get(m, None), ms=3.5, mec="white", mew=0.5) if m in COLOR else {}
        ax.plot(res, mu, **st, **mk, label=name(m) if m in COLOR else None)
    ax.axvline(224, color=INK2, lw=0.6, ls=":")
    ax.set_xticks(res); ax.set_xticklabels([str(r) for r in res])
    ax.set_ylim(55, 81.5)
    ax.grid(True, axis="y")
    ax.set_xlabel("test resolution (px), trained at 224")
    ax.set_ylabel("test top-1 (%)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.45, -0.24), fontsize=6,
              columnspacing=0.9, handlelength=2.0)
    save(fig, "fig_resolution")
    return f"{len(runs)} methods, {rmin}-{rmax}"


# style of the two-panel main-paper figures: sans-serif chart text, boxed panels, dotted grid on both axes,
# panels wider than tall, shared legend on top, "(a) ..." sub-captions below in the paper's serif font
PAIR_STYLE = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
              "axes.spines.top": True, "axes.spines.right": True, "axes.edgecolor": "#333333", "axes.linewidth": 0.6,
              "xtick.direction": "out", "ytick.direction": "out", "xtick.color": "#333333", "ytick.color": "#333333",
              "grid.color": "#b5b5b5", "grid.linestyle": ":", "grid.linewidth": 0.6}
PAIR_FS = 6.5


def _pair_axes(ax, xlabel, ylabel):
    ax.grid(True, which="major", axis="both")
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=PAIR_FS, labelpad=1.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=PAIR_FS, labelpad=1.5)
    ax.tick_params(labelsize=PAIR_FS - 0.5, length=2, width=0.5, pad=1.5)
    ax.set_box_aspect(0.8)                                            # wider than tall


def _pair_finish(fig, axes, captions, handles, labels, ncol, stem, legend_h=0.12):
    fig.tight_layout(rect=(0, 0.1, 1, 1 - legend_h), w_pad=1.2, pad=0.2)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    top = max(ax.get_tightbbox(r).transformed(inv).y1 for ax in axes)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, top + 0.005),
               fontsize=PAIR_FS - 0.5, columnspacing=1.0, handlelength=1.8, handletextpad=0.5, borderaxespad=0.0)
    for ax, cap in zip(axes, captions):
        bb = ax.get_tightbbox(r).transformed(inv)
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, bb.y0 - 0.015, cap, ha="center", va="top", fontsize=7.5,
                 family="serif", math_fontfamily="stix")
    save(fig, stem)                                                   # bbox "tight" crops the unused margin


def fig_curves_res(group="in100_deit_small", lo=70, rmin=128, rmax=448):
    """Main paper, one column: (a) validation curves of the last 30 epochs and (b) test top-1 across resolutions,
    side by side with one shared legend (replaces fig_curves_main + fig_resolution in the main text)."""
    runs = finished(group)
    if not runs:
        return "no runs"
    allr = load_group(os.path.join(RES, group))
    if "abl_no_gates" in allr:
        runs["abl_no_gates"] = allr["abl_no_gates"]
    order = [m for m in runs if m not in COLOR and m != "abl_no_gates"] + ["abl_no_gates"] + \
        [m for m in HIGHLIGHT[::-1] if m in runs]
    main_runs = _main_runs()
    rr = finished(group)
    if "abl_no_gates" in main_runs:
        rr = dict(rr, abl_no_gates=main_runs["abl_no_gates"])
    res = [r for r in sorted({int(r) for v in rr.values() for f in v.values() for r in f.get("multires", {})}) if rmin <= r <= rmax]
    lw = lambda m: 1.5 if m == "viper" else (1.0 if m in COLOR else 0.6)
    with plt.rc_context(PAIR_STYLE):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 2.05))
        for m in order:                                           # (a) validation curves
            h = histories(group, m, runs[m].keys())
            if h.size == 0:
                continue
            ep = np.arange(1, h.shape[1] + 1)
            mu, sd = h.mean(0), h.std(0)
            sel = ep >= lo
            if m == "abl_no_gates":
                a1.plot(ep[sel], mu[sel], color=COLOR["viper"], ls="--", lw=1.0, zorder=4)
                continue
            st = dict(style(m)); st["lw"] = lw(m)
            a1.plot(ep[sel], mu[sel], **st)
            if h.shape[0] > 1 and m in COLOR:
                a1.fill_between(ep[sel], (mu - sd)[sel], (mu + sd)[sel], color=st["color"], alpha=0.15, lw=0)
        a1.set_ylim(73, 82.5); a1.set_xticks([70, 80, 90, 100]); a1.set_yticks([74, 76, 78, 80, 82])
        _pair_axes(a1, "epoch", "val top-1 (%)")
        for m in order:                                           # (b) resolution
            if m not in rr:
                continue
            mu = np.array([[f["multires"][str(r)]["extrapolate"]["top1"] for r in res] for f in rr[m].values()]).mean(0)
            if m == "abl_no_gates":
                a2.plot(res, mu, color=COLOR["viper"], ls="--", lw=1.0, marker="o", ms=2.4, mfc="white",
                        mec=COLOR["viper"], mew=0.6, label="ViPER w/o gates", zorder=4)
                continue
            st = dict(style(m)); st["lw"] = lw(m)
            mk = dict(marker=MARKER.get(m, None), ms=2.6, mec="white", mew=0.3) if m in COLOR else {}
            a2.plot(res, mu, **st, **mk, label=name(m) if m in COLOR else None)
        a2.axvline(224, color="#555555", lw=0.5, ls="--")
        a2.set_xticks([128, 224, 320, 448]); a2.set_ylim(55, 81.5); a2.set_yticks([55, 60, 65, 70, 75, 80])
        _pair_axes(a2, "test resolution (px)", "test top-1 (%)")
        h, l = a2.get_legend_handles_labels()
        _pair_finish(fig, (a1, a2), ("(a) Validation, last 30 epochs", "(b) Test, across resolutions"),
                     h[::-1], l[::-1], 3, "fig_curves_res", legend_h=0.13)
    return "ok"


PILOT_GROUPS = [  # (label, [(run, display)]): every ViPER variant sits directly below its matched control
    ("rope", [("rope_mixed", "RoPE-Mixed"), ("viper_theta_mixed", "ViPER (= RoPE-Mixed + gates)"),
              ("viper_fgate", "ViPER, feature-wavelet gates")]),
    ("string", [("string_cayley", "STRING")]),
    ("lepe", [("lepe", "LePE only"), ("rope_mixed_lepe", "RoPE-Mixed + LePE"),
              ("viper_lepe", "ViPER + LePE")]),
    ("wv", [("rope_mixed_wv", "RoPE-Mixed + WaveLePE"), ("viper_wv", "ViPER + WaveLePE"),
            ("viper_wvg", "ViPER + gated WaveLePE"), ("viper_wvg1", "ViPER + gated WaveLePE (J=1)")]),
]


def fig_pilots(group="pilot_in100_30ep"):
    runs = load_group(os.path.join(RES, group))
    rows = [(g, r, d) for g, items in PILOT_GROUPS for r, d in items if r in runs]
    if len(rows) < 3:
        return "too few pilots"
    fig, ax = plt.subplots(figsize=(COL_W, 0.2 * len(rows) + 0.6))
    y = np.arange(len(rows))[::-1]
    ref = np.mean([v["best_val_top1"] for v in runs["rope_mixed"].values()]) if "rope_mixed" in runs else None
    for yi, (g, r, d) in zip(y, rows):
        vals = np.array([v["best_val_top1"] for v in runs[r].values()])
        ours = r.startswith("viper")
        c = COLOR["viper"] if ours else INK2
        ax.hlines(yi, vals.min(), vals.max(), color=c, lw=1.0)
        ax.plot(vals, [yi] * len(vals), "o", ms=3.2, mfc="white", mec=c, mew=1.0, zorder=4)
        ax.plot(vals.mean(), yi, "|", ms=9, mew=2.0, color=c, zorder=5)
        ax.text(1.005, yi, f"{vals.mean():.2f}" + ("" if len(vals) > 1 else "$^1$"), transform=ax.get_yaxis_transform(),
                va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([d for _, _, d in rows])
    lo = min(np.mean([v["best_val_top1"] for v in runs[r].values()]) for _, r, _ in rows)
    hi = max(max(v["best_val_top1"] for v in runs[r].values()) for _, r, _ in rows)
    ax.set_xlim(lo - 0.8, hi + 0.8)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=0.6, ls=":")
    # dividers between groups of matched controls
    groups = [g for g, _, _ in rows]
    for k in range(1, len(rows)):
        if groups[k] != groups[k - 1]:
            ax.axhline((y[k] + y[k - 1]) / 2, color=INK2, lw=0.5)
    ax.set_xlabel("val top-1 (%), 30-epoch pilots")
    ax.set_ylim(-0.7, len(rows) - 0.2)
    ax.grid(True, axis="x")
    save(fig, "fig_pilots")
    return f"{len(rows)} pilot variants"


TWOPATH = [  # (group, run, label): baselines, then each control directly above its ViPER variant
    ("in100_deit_small", "rope_mixed", "RoPE-Mixed"), ("in100_deit_small", "abl_no_value", "ViPER, score path only"),
    ("in100_deit_small", "string_cayley", "STRING"), ("in100_deit_small", "lepe", "LePE only"),
    ("in100_deit_small", "abl_no_gates", "ViPER w/o gates"), ("in100_deit_small", "viper", "ViPER (ours)"),
]


def fig_twopath():
    """(a) test top-1 of every seed, each ViPER variant below its control; (b) validation curves (mean over
    finished seeds) of the two-path models and the strongest single-path baselines."""
    groups = {g: load_group(os.path.join(RES, g), five=True) for g in {g for g, _, _ in TWOPATH}}
    rows = [(g, r, d) for g, r, d in TWOPATH if r in groups[g]]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXT_W, 2.5), gridspec_kw=dict(width_ratios=[1.05, 1]))
    y = np.arange(len(rows))[::-1]
    for yi, (g, r, d) in zip(y, rows):
        vals = np.array([v["test"]["top1"] for v in groups[g][r].values()])
        c = COLOR["viper"] if r.startswith("viper") else INK2
        ax.hlines(yi, vals.min(), vals.max(), color=c, lw=1.0)
        ax.plot(vals, [yi] * len(vals), "o", ms=3.2, mfc="white", mec=c, mew=1.0, zorder=4)
        ax.plot(vals.mean(), yi, "|", ms=9, mew=2.0, color=c, zorder=5)
        ax.text(1.01, yi, f"{vals.mean():.2f}" + (f"$^{len(vals)}$" if len(vals) < 5 else ""),
                transform=ax.get_yaxis_transform(), va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([d for _, _, d in rows])
    ax.grid(True, axis="x")
    ax.set_xlabel("test top-1 (%), 100 epochs")
    ax.set_title("(a) test accuracy per seed", loc="left")
    curves = [("in100_deit_small", "string_cayley", "STRING", COLOR["string_cayley"], "-"),
              ("in100_deit_small", "lepe", "LePE", COLOR["lepe"], "-"),
              ("in100_deit_small", "abl_no_gates", "ViPER w/o gates", COLOR["rope_mixed"], "--"),
              ("in100_deit_small", "viper", "ViPER (ours)", COLOR["viper"], "-")]
    for g, r, d, c, ls in curves:
        if r not in groups[g]:
            continue
        h = histories(g, r, groups[g][r].keys())
        if h.size == 0:
            continue
        ep = np.arange(1, h.shape[1] + 1)
        sel = ep >= 70
        bx.plot(ep[sel], h.mean(0)[sel], color=c, ls=ls, lw=1.8 if r.startswith("viper") else 1.2,
                label=d + (f" ({h.shape[0]})" if h.shape[0] > 1 else ""))
    bx.grid(True, axis="y")
    bx.set_xlabel("epoch")
    bx.set_ylabel("val top-1 (%), 5% train holdout")
    bx.set_title("(b) last 30 epochs", loc="left")
    bx.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_twopath")
    return f"{len(rows)} rows"


def _plot_gates(ax, G):
    """Mean gate per band (seeds as dots) and Spearman rho between a token's gate and its fine wavelet energy."""
    nb = len(next(iter(G.values()))["mean"][0])
    bands = np.arange(nb)
    means = np.array([np.mean(g["mean"], 0) for g in G.values()])            # (seeds, nb)
    rhos = np.array([np.mean(g["rho_fine"], 0) for g in G.values()])
    ax.bar(bands - 0.18, means.mean(0), 0.36, color=COLOR["viper"], label="mean gate")
    ax.scatter(np.repeat(bands - 0.18, len(means)), means.T.flatten(), s=6, color=INK, zorder=4)
    ax.axhline(1.0, color=INK2, lw=0.6, ls=":")
    ax.set_ylabel("mean gate $g$")
    ax2 = ax.twinx()
    ax2.bar(bands + 0.18, rhos.mean(0), 0.36, color="#eda100", label=r"$\rho$(gate, fine energy)")
    ax2.scatter(np.repeat(bands + 0.18, len(rhos)), rhos.T.flatten(), s=6, color=INK, zorder=4)
    ax2.axhline(0, color=INK2, lw=0.6)
    ax2.set_ylabel(r"Spearman $\rho$ with fine energy")
    ax2.spines["right"].set_visible(True)
    ax.set_xticks(bands)
    ax.set_xticklabels([f"{b + 1}" for b in bands])
    ax.set_xlabel("band (1 = finest, 4 = coarsest)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.set_ylim(0, 2.2); ax2.set_ylim(-0.3, 0.95)
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=5.5, loc="upper center", ncol=2, columnspacing=0.8, handlelength=1.2)


def _plot_attn(ax, A):
    """Mean attention distance per layer."""
    style_ = {"viper": (COLOR["viper"], "-", "ViPER"), "abl_no_gates": (COLOR["rope_mixed"], "--", "ViPER w/o gates"),
              "rope_mixed": (COLOR["rope_mixed"], "-", "RoPE-Mixed"), "string_cayley": (COLOR["string_cayley"], "-", "STRING"),
              "lepe": (COLOR["lepe"], "-", "LePE"), "lookhere": (COLOR["lookhere"], "-", "LookHere"),
              "carope": (GREY, "-", "CARoPE")}
    for k, v in A.items():
        m = k.rsplit("_s", 1)[0]
        c, ls, lab = style_.get(m, (GREY, "-", m))
        ax.plot(np.arange(1, len(v) + 1), v, color=c, ls=ls, lw=1.8 if m == "viper" else 1.2, label=lab, marker="o", ms=2.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean attention distance (patches)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=5.5, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.22), columnspacing=0.8, handlelength=1.6)
    ax.set_xticks([1, 4, 8, 12])


def _plot_pareto(ax, p):
    """Accuracy vs. inference throughput for every encoding."""
    E = json.load(open(p))["results"]
    runs = _main_runs()
    for m, e in E.items():
        if m not in runs or m not in PE_INFO:                                # ViPER w/o gates not shown
            continue
        acc = np.mean([f["test"]["top1"] for f in runs[m].values()])
        if m == "abl_no_gates":                                              # matched control: hollow ViPER marker
            ax.plot(e["infer_img_s"], acc, "o", ms=4.5, mfc="white", mec=COLOR["viper"], mew=1.0, zorder=4)
        else:
            ax.plot(e["infer_img_s"], acc, MARKER.get(m, "o"), ms=7 if m == "viper" else 4.5, color=COLOR.get(m, GREY),
                    zorder=5 if m == "viper" else 3)
        off = {"viper": (7, -2), "string_cayley": (-42, -2), "lepe": (-8, 6),
               "lookhere": (-4, 6), "rope_mixed": (4, -10), "liere": (-6, 6), "comrope_ld": (4, -9), "none": (-28, 5)}
        if m in off:
            ax.annotate({"string_cayley": "STRING", "comrope_ld": "ComRoPE", "abl_no_gates": "ViPER w/o gates"}.get(
                            m, name(m).replace(" (ours)", "")),
                        (e["infer_img_s"], acc), fontsize=7.5 if m == "viper" else 7, xytext=off[m], textcoords="offset points", color=INK)
    ax.set_xlabel("inference throughput (images/s, DeiT-S, bs 128)")
    ax.set_ylabel("ImageNet-100 top-1 (%)")
    ax.grid(True)


def fig_analysis():
    """Two column-width panels (placed side by side in a figure*, aligned with the text columns):
    fig_analysis_a: accuracy vs. inference throughput; fig_analysis_b: gate source vs. test resolution."""
    ep = os.path.join(RES, "efficiency_deit_small.json")
    if not os.path.exists(ep):
        return "no efficiency results"
    fig, px = plt.subplots(figsize=(COL_W, 2.05))
    _plot_pareto(px, ep)
    px.set_xlabel("inference throughput (images/s)")
    save(fig, "fig_analysis_a")
    runs = _gate_runs()
    if all(m in runs for m in ("abl_no_gates", "viper", "abl_static_gate", "abl_rgb_gate")):
        fig, cx = plt.subplots(figsize=(COL_W, 2.05))
        _plot_gatectl(cx, runs, rmin=160, rmax=384, legend_loc="upper center")
        n = len(set(runs["abl_static_gate"]) & set(runs["abl_rgb_gate"]))
        save(fig, "fig_analysis_b")
    return "ok"


def fig_cost_pair():
    """Main paper, one column: ImageNet-100 top-1 vs. (a) inference throughput and (b) positional parameters,
    every encoding as a point; the five strongest baselines, ViPER and its control in colour."""
    ep = os.path.join(RES, "efficiency_deit_small.json")
    if not os.path.exists(ep):
        return "no efficiency results"
    E = json.load(open(ep))["results"]
    runs = _main_runs()
    pts = [(m, e, np.mean([f["test"]["top1"] for f in runs[m].values()])) for m, e in E.items()
           if m in runs and (m in PE_INFO or m == "abl_no_gates")]
    zero = 0.3                                                         # 0-parameter encodings drawn at 0.3K
    with plt.rc_context(PAIR_STYLE):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 2.0))
        for ax, xf in ((a1, lambda e: e["infer_img_s"]), (a2, lambda e: max(e["pe_params"] / 1e3, zero))):
            for m, e, acc in sorted(pts, key=lambda t: (t[0] in COLOR, t[0] == "viper")):
                if m == "abl_no_gates":
                    ax.plot(xf(e), acc, "o", ms=4, mfc="white", mec=COLOR["viper"], mew=0.9, zorder=4,
                            label="ViPER w/o gates")
                elif m in COLOR:
                    ax.plot(xf(e), acc, MARKER.get(m, "o"), ms=5.5 if m == "viper" else 4, color=COLOR[m],
                            mec="white", mew=0.4, zorder=5 if m == "viper" else 3, label=name(m))
                else:
                    ax.plot(xf(e), acc, "o", ms=3, color=GREY, alpha=0.8, zorder=1,
                            label="other encodings")
            ax.set_ylim(65.5, 80.5); ax.set_yticks([66, 70, 74, 78])
        a1.set_xticks([1000, 2000, 3000, 4000]); a1.set_xticklabels(["1k", "2k", "3k", "4k"])
        a2.set_xscale("log"); a2.set_xlim(0.2, 5000)
        a2.set_xticks([zero, 1, 10, 100, 1000]); a2.set_xticklabels(["0", "1K", "10K", "100K", "1M"])
        a2.minorticks_off()
        _pair_axes(a1, "inference throughput (img/s)", "ImageNet-100 top-1 (%)")
        _pair_axes(a2, "positional parameters", None)
        h, l = a1.get_legend_handles_labels()
        uniq = {}
        for hh, ll in zip(h, l):
            uniq.setdefault(ll, hh)
        order = ["ViPER (ours)", "STRING", "LePE", "LookHere", "RoPE-Mixed", "ViPER w/o gates", "other encodings"]
        keys = [k for k in order if k in uniq]
        _pair_finish(fig, (a1, a2), ("(a) Accuracy vs. speed", "(b) Accuracy vs. PE parameters"),
                     [uniq[k] for k in keys], keys, 4, "fig_cost_pair", legend_h=0.13)
    return "ok"


def fig_gatemaps_main():
    """Column-width gate maps for the main paper (3 test images)."""
    if not os.path.exists(os.path.join(RES, "analysis", "gate_maps.npz")):
        return "no gate maps"
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    _plot_gatemaps(fig, ax, rows=3, title=False, cbar=True)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.06)
    save(fig, "fig_gatemaps_main")
    return "ok"


def fig_supp_gatebars():
    """Mean gate per band and its correlation with fine wavelet energy (was Fig. 4b of the main paper)."""
    gp = os.path.join(RES, "analysis", "gates.json")
    if not os.path.exists(gp):
        return "no gates"
    G = json.load(open(gp))
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    _plot_gates(ax, G)
    save(fig, "fig_supp_gatebars")
    return "ok"


def fig_supp_attn():
    """Mean attention distance per layer (seed 0)."""
    ap = os.path.join(RES, "analysis", "attn_distance.json")
    if not os.path.exists(ap):
        return "no attention distances"
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    _plot_attn(ax, json.load(open(ap)))
    save(fig, "fig_supp_attn")
    return "ok"


def _plot_gatemaps(fig, ax, rows=3, title=True, cbar=False):
    """Replace axis `ax` by a rows x 4 grid: image, fine wavelet energy, gate of the finest and coarsest band.
    Each gate column uses one colour scale for all images (shown as a colour bar when cbar=True)."""
    z = np.load(os.path.join(RES, "analysis", "gate_maps.npz"))
    img, g, fine = z["img"], z["gates"], z["fine"]
    mean, std = np.array([0.485, 0.456, 0.406])[:, None, None], np.array([0.229, 0.224, 0.225])[:, None, None]
    gm_all = [g[i].mean(0) for i in range(rows)]
    lims = {b: (np.percentile([x[:, b] for x in gm_all], 2), np.percentile([x[:, b] for x in gm_all], 98)) for b in (0, -1)}
    sub = ax.get_subplotspec().subgridspec(rows + (1 if cbar else 0), 4, wspace=0.03, hspace=0.03,
                                           height_ratios=[1] * rows + ([0.07] if cbar else []))
    ax.remove()
    cols = ["image", "texture", "fine gate", "coarse gate"]
    ims = {}
    for i in range(rows):
        im = np.clip((img[i] * std + mean).transpose(1, 2, 0), 0, 1)
        gm = gm_all[i]
        for j, pnl in enumerate([im, fine[i], gm[:, 0].reshape(14, 14), gm[:, -1].reshape(14, 14)]):
            a_ = fig.add_subplot(sub[i, j])
            kw = {}
            if j >= 2:
                kw = dict(vmin=lims[0 if j == 2 else -1][0], vmax=lims[0 if j == 2 else -1][1])
            ims[j] = a_.imshow(pnl, cmap=None if j == 0 else ("magma" if j == 1 else "viridis"), **kw)
            a_.set_xticks([]); a_.set_yticks([])
            for sp in a_.spines.values():
                sp.set_visible(False)
            if i == 0:
                a_.set_title(cols[j], fontsize=6 if title else 7.5, pad=1.5)
            if title and i == 0 and j == 0:
                a_.text(0, 1.28, "(b) gates follow image texture", transform=a_.transAxes, fontsize=plt.rcParams["axes.titlesize"])
    if cbar:
        ca = fig.add_subplot(sub[rows, 1]); cb = fig.colorbar(ims[1], cax=ca, orientation="horizontal")
        cb.set_ticks([]); ca.set_xlabel("low  →  high", fontsize=6, labelpad=1)
        for j in (2, 3):
            ca = fig.add_subplot(sub[rows, j])
            cb = fig.colorbar(ims[j], cax=ca, orientation="horizontal")
            cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
            lo, hi = lims[0 if j == 2 else -1]
            cb.set_ticks([round(lo + 0.2 * (hi - lo), 2), round(hi - 0.2 * (hi - lo), 2)])


def fig_gatemaps():
    p = os.path.join(RES, "analysis", "gate_maps.npz")
    if not os.path.exists(p):
        return "no gate maps"
    z = np.load(p)
    img, g, fine = z["img"], z["gates"], z["fine"]                            # g: (B, L, N, nb)
    mean, std = np.array([0.485, 0.456, 0.406])[:, None, None], np.array([0.229, 0.224, 0.225])[:, None, None]
    B = img.shape[0]
    cols = ["image", "fine wavelet energy", "gate, finest band", "gate, coarsest band"]
    fig, axes = plt.subplots(B, 4, figsize=(COL_W, 0.85 * B + 0.2))
    for i in range(B):
        im = np.clip((img[i] * std + mean).transpose(1, 2, 0), 0, 1)
        gm = g[i].mean(0)                                                     # mean over layers: (N, nb)
        panels = [im, fine[i], gm[:, 0].reshape(14, 14), gm[:, -1].reshape(14, 14)]
        for j, (a_, pnl) in enumerate(zip(axes[i], panels)):
            a_.imshow(pnl, cmap=None if j == 0 else ("magma" if j == 1 else "viridis"))
            a_.set_xticks([]); a_.set_yticks([])
            if i == 0:
                a_.set_title(cols[j], fontsize=6)
    fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
    save(fig, "fig_gatemaps")
    return "ok"


# --------------------------------------------------------------------------- supplementary figures
def _main_runs():
    """seeds 0-2 for every method (comparisons with the baselines)"""
    return load_group(os.path.join(RES, "in100_deit_small"))


def _gate_runs():
    """ViPER and its gate controls over seeds 0-4 (the paired gate comparison)"""
    return load_group(os.path.join(RES, "in100_deit_small"), five=True)


def fig_supp_paired():
    """Per-seed gain of ViPER over its matched control (ViPER w/o gates) at every test resolution."""
    runs = _gate_runs()
    if "viper" not in runs or "abl_no_gates" not in runs:
        return "no runs"
    v, c = runs["viper"], runs["abl_no_gates"]
    seeds = sorted(set(v) & set(c))
    res = sorted({int(r) for f in v.values() for r in f["multires"]})
    get = lambda f, r: f["test"]["top1"] if r == 224 else f["multires"][str(r)]["extrapolate"]["top1"]
    d = np.array([[get(v[s], r) - get(c[s], r) for r in res] for s in seeds])
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    x = np.arange(len(res))
    for i, s in enumerate(seeds):
        ax.plot(x + (i - (len(seeds) - 1) / 2) * 0.07, d[i], "o", ms=3, mfc="white", mec=COLOR["viper"], mew=0.9)
    ax.bar(x, d.mean(0), 0.55, color=COLOR["viper"], alpha=0.35, zorder=0)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axvline(res.index(224), color=INK2, lw=0.6, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(res)
    ax.set_xlabel("test resolution (px)")
    ax.set_ylabel("$\\Delta$ top-1 vs. control")
    ax.grid(True, axis="y")
    save(fig, "fig_supp_paired")
    return f"{len(seeds)} seeds"


def fig_supp_gates():
    """(a) mean gate per layer and band; (b) Spearman correlation with fine wavelet energy per layer and band."""
    p = os.path.join(RES, "analysis", "gates.json")
    if not os.path.exists(p):
        return "no gates"
    G = json.load(open(p))
    mean = np.mean([np.array(g["mean"]) for g in G.values()], 0)                # (L, nb)
    rho = np.mean([np.array(g["rho_fine"]) for g in G.values()], 0)
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9))
    for ax, a, cm, t, lim in ((axes[0], mean, "viridis", "(a) mean gate", None),
                              (axes[1], rho, "RdBu_r", r"(b) $\rho$ with fine energy", 0.6)):
        im = ax.imshow(a.T, aspect="auto", cmap=cm, origin="lower",
                       vmin=-lim if lim else None, vmax=lim if lim else None)
        ax.set_xlabel("layer"); ax.set_ylabel("band")
        ax.set_xticks(range(0, a.shape[0], 2)); ax.set_xticklabels(range(1, a.shape[0] + 1, 2))
        ax.set_yticks(range(a.shape[1])); ax.set_yticklabels(range(1, a.shape[1] + 1))
        ax.set_title(t, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=6)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_supp_gates")
    return "ok"


def fig_supp_probe():
    files = sorted(glob.glob(os.path.join(RES, "analysis", "probe_*.json")))
    if not files:
        return "no probes"
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for f in files:
        m = os.path.basename(f)[6:-5]
        P = json.load(open(f)); L = sorted(P, key=int)
        y = [(P[l]["r2_x"] + P[l]["r2_y"]) / 2 for l in L]
        lab = {"abl_no_gates": "ViPER w/o gates", "learned": "APE (learned)", "none": "No PE"}.get(m, name(m).replace(" (ours)", ""))
        col = {"abl_no_gates": COLOR["rope_mixed"], "learned": "#8a6d3b", "none": GREY}.get(m, COLOR.get(m, GREY))
        ax.plot([int(l) for l in L], y, marker="o", ms=2.5, color=col,
                ls={"abl_no_gates": "--", "none": ":"}.get(m, "-"), lw=1.8 if m == "viper" else 1.1, label=lab)
    ax.set_xlabel("layer (0 = patch embedding + PE)")
    ax.set_ylabel("$R^2$ of linear coordinate probe")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=5.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    save(fig, "fig_supp_probe")
    return "ok"


def _plot_gatectl(ax, runs, legend_loc="upper right", rmin=0, rmax=10**4):
    """Gain over the no-gate control of ViPER and of the two gate-source controls (static, RGB), per resolution."""
    ms = [("viper", "ViPER (wavelet gates)", COLOR["viper"], "o"), ("abl_rgb_gate", "RGB gates", INK2, "s"),
          ("abl_static_gate", "static gates", GREY, "^")]
    c = runs["abl_no_gates"]
    res = sorted({int(r) for f in c.values() for r in f["multires"] if rmin <= int(r) <= rmax})
    get = lambda f, r: f["test"]["top1"] if r == 224 else f["multires"][str(r)]["extrapolate"]["top1"]
    x = np.arange(len(res))
    for k, (m, lab, col, mk) in enumerate(ms):
        seeds = sorted(set(runs[m]) & set(c))
        d = np.array([[get(runs[m][s], r) - get(c[s], r) for r in res] for s in seeds])
        se = d.std(0, ddof=1) / np.sqrt(len(seeds))
        ax.errorbar(x, d.mean(0), yerr=se, fmt=mk + "-", ms=3.5, lw=1.0, capsize=1.5, color=col,
                    mfc="white" if m != "viper" else col, label=lab, zorder=5 - k)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axvline(res.index(224), color=INK2, lw=0.6, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(res)
    ax.set_xlabel("test resolution (px)")
    ax.set_ylabel("$\\Delta$ top-1 vs. no gates")
    ax.grid(True, axis="y")
    ax.legend(fontsize=6, frameon=False, loc=legend_loc, ncol=1)


def fig_supp_gatectl():
    """All test resolutions of the gate-source comparison (the main figure shows 160 and above)."""
    runs = _gate_runs()
    if any(m not in runs for m in ("abl_no_gates", "viper", "abl_static_gate", "abl_rgb_gate")):
        return "no runs"
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    _plot_gatectl(ax, runs)
    save(fig, "fig_supp_gatectl")
    return "ok"


def fig_supp_texture():
    """Gain over the no-gate control per quintile of image texture energy (results/analysis/texture_gain.json)."""
    p = os.path.join(RES, "analysis", "texture_gain.json")
    if not os.path.exists(p):
        return "no texture analysis"
    T = json.load(open(p))
    ms = [("viper", "ViPER (wavelet gates)", COLOR["viper"], "o"), ("abl_rgb_gate", "RGB gates", INK2, "s"),
          ("abl_static_gate", "static gates", GREY, "^")]
    c = T["acc"]["abl_no_gates"]
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    x = np.arange(5)
    for k, (m, lab, col, mk) in enumerate(ms):
        seeds = sorted(set(T["acc"][m]) & set(c))
        d = np.array([np.array(T["acc"][m][s]["224"]) - np.array(c[s]["224"]) for s in seeds])
        ax.errorbar(x, d.mean(0), yerr=d.std(0, ddof=1) / np.sqrt(len(seeds)), fmt=mk + "-", ms=3.5,
                    lw=1.0, capsize=1.5, color=col, mfc="white" if m != "viper" else col, label=lab, zorder=5 - k)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(["1\n(smoothest)", "2", "3", "4", "5\n(most textured)"])
    ax.set_xlabel("quintile of fine-scale wavelet energy of the test image")
    ax.set_ylabel("$\\Delta$ top-1 vs. no gates")
    ax.grid(True, axis="y")
    ax.legend(fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.0, handlelength=1.6)
    save(fig, "fig_supp_texture")
    return f"{len(T['n_per_bin'])} bins"


def fig_gate_pair():
    """Main paper, one column: (a) gate source per test resolution (160-384) and (b) gain by image texture,
    both as top-1 gain over ViPER w/o gates, with one shared legend (replaces fig_analysis_b + fig_supp_texture)."""
    runs = _gate_runs()
    p = os.path.join(RES, "analysis", "texture_gain.json")
    if any(m not in runs for m in ("abl_no_gates", "viper", "abl_static_gate", "abl_rgb_gate")) or not os.path.exists(p):
        return "missing runs or texture analysis"
    T = json.load(open(p))
    ms = [("viper", "ViPER (wavelet gates)", COLOR["viper"], "o"), ("abl_rgb_gate", "RGB gates", INK2, "s"),
          ("abl_static_gate", "static gates", GREY, "^")]
    c = runs["abl_no_gates"]
    res = sorted({int(r) for f in c.values() for r in f["multires"] if 160 <= int(r) <= 384})
    get = lambda f, r: f["test"]["top1"] if r == 224 else f["multires"][str(r)]["extrapolate"]["top1"]
    eb = dict(ms=3.0, lw=1.0, capsize=1.5, elinewidth=0.7, capthick=0.7)
    with plt.rc_context(PAIR_STYLE):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.95))
        x = np.arange(len(res))
        for k, (m, lab, col, mk) in enumerate(ms):                # (a) gate source by resolution
            seeds = sorted(set(runs[m]) & set(c))
            d = np.array([[get(runs[m][s], r) - get(c[s], r) for r in res] for s in seeds])
            a1.errorbar(x, d.mean(0), yerr=d.std(0, ddof=1) / np.sqrt(len(seeds)), fmt=mk + "-", color=col,
                        mfc="white" if m != "viper" else col, label=lab, zorder=5 - k, **eb)
        a1.axvline(res.index(224), color="#555555", lw=0.5, ls="--")
        a1.set_xticks(x); a1.set_xticklabels(res)
        ct = T["acc"]["abl_no_gates"]
        for k, (m, lab, col, mk) in enumerate(ms):                # (b) gain by texture quintile
            seeds = sorted(set(T["acc"][m]) & set(ct))
            d = np.array([np.array(T["acc"][m][s]["224"]) - np.array(ct[s]["224"]) for s in seeds])
            a2.errorbar(np.arange(5), d.mean(0), yerr=d.std(0, ddof=1) / np.sqrt(len(seeds)), fmt=mk + "-",
                        color=col, mfc="white" if m != "viper" else col, zorder=5 - k, **eb)
        a2.set_xticks(range(5)); a2.set_xticklabels(["1", "2", "3", "4", "5"])
        for ax in (a1, a2):
            ax.axhline(0, color="#333333", lw=0.6)
        _pair_axes(a1, "test resolution (px)", "$\\Delta$ top-1 vs. no gates (%)")
        _pair_axes(a2, "texture quintile (1: smooth)", None)
        h, l = a1.get_legend_handles_labels()
        _pair_finish(fig, (a1, a2), ("(a) Gate source", "(b) Gain by image texture"), h, l, 3, "fig_gate_pair",
                     legend_h=0.08)
    return "ok"


def fig_supp_robust():
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_tables import load_robust, ROBUST_SETS
    R = load_robust()
    show = [m for m in ("none", "rope_mixed", "string_cayley", "lepe", "lookhere", "abl_no_gates", "viper") if m in R]
    if len(show) < 3:
        return "no robustness"
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    w = 0.8 / len(show)
    for i, m in enumerate(show):
        vals = [np.mean(list(R[m][k].values())) for k, _ in ROBUST_SETS]
        lab = {"abl_no_gates": "ViPER w/o gates", "none": "No PE"}.get(m, name(m).replace(" (ours)", ""))
        ax.bar(np.arange(len(ROBUST_SETS)) + (i - (len(show) - 1) / 2) * w, vals, w,
               color=COLOR.get(m, "#c9c7c1" if m == "abl_no_gates" else GREY), label=lab,
               hatch="//" if m == "abl_no_gates" else None, edgecolor="white", lw=0.3)
    ax.set_xticks(range(len(ROBUST_SETS))); ax.set_xticklabels([l for _, l in ROBUST_SETS])
    ax.set_ylabel("top-1 (%)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=5.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    save(fig, "fig_supp_robust")
    return "ok"


def fig_supp_kernel():
    """Effective positional kernel of one ViPER layer for the most textured and the smoothest query of a few test
    images (scripts/analysis.py kernel), next to the kernel with all gates at 1 (the matched control)."""
    kp = os.path.join(RES, "analysis", "kernel.json")
    if not os.path.exists(kp):
        return "no kernel"
    K = json.load(open(kp))
    R = K["R"]
    diff = {l: np.abs(np.array(v["textured"]) - np.array(v["smooth"])).mean() for l, v in K["layers"].items()}
    l = max(diff, key=diff.get)
    L = K["layers"][l]
    o, t, sm = (np.array(L[k]) for k in ("ones", "textured", "smooth"))
    fig, axs = plt.subplots(1, 5, figsize=(TEXT_W, 1.65), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.25]))
    vmin, vmax = min(o.min(), t.min(), sm.min()), max(o.max(), t.max(), sm.max())
    ext = [-R - .5, R + .5, R + .5, -R - .5]
    for ax, z, ttl in ((axs[0], o, "all gates $=1$"), (axs[1], t, "textured query"), (axs[2], sm, "smooth query")):
        im = ax.imshow(z, cmap="viridis", vmin=vmin, vmax=vmax, extent=ext)
        ax.set_title(ttl); ax.set_xticks([-10, 0, 10]); ax.set_yticks([-10, 0, 10])
    axs[0].set_ylabel(r"$\Delta y$ (patches)")
    for ax in axs[:4]:
        ax.set_xlabel(r"$\Delta x$")
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.03)
    d = t / t[R, R] - sm / sm[R, R]                                        # shape difference (centre-normalised)
    lim = np.abs(d).max()
    im2 = axs[3].imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext)
    axs[3].set_title("shape: textured $-$ smooth"); axs[3].set_xticks([-10, 0, 10]); axs[3].set_yticks([-10, 0, 10])
    fig.colorbar(im2, ax=axs[3], fraction=0.046, pad=0.03)
    r = np.arange(0, R + 1)
    for z, lab, c, ls in ((o, "gates $=1$", GREY, "--"), (t, "textured", COLOR["viper"], "-"), (sm, "smooth", COLOR["rope_mixed"], "-")):
        prof = np.array([np.mean([z[R + dy, R + dx] for dy in range(-R, R + 1) for dx in range(-R, R + 1)
                                  if round(np.hypot(dx, dy)) == k]) for k in r])
        axs[4].plot(r, prof / prof[0], color=c, ls=ls, label=lab)
    axs[4].set_xlabel(r"$|\Delta|$ (patches)"); axs[4].set_ylabel("kernel / kernel(0)")
    axs[4].set_ylim(0.35, 1.45); axs[4].set_yticks([0.4, 0.6, 0.8, 1.0]); axs[4].legend(frameon=False, loc="upper right", fontsize=6, handlelength=1.2); axs[4].grid(True)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_supp_kernel")
    return f"layer {int(l) + 1}"


def fig_supp_attntex():
    """Mean attention distance of textured minus smooth queries per layer: ViPER vs its control (3 seeds)."""
    ap = os.path.join(RES, "analysis", "attn_texture.json")
    if not os.path.exists(ap):
        return "no attn_texture"
    A = json.load(open(ap))
    fig, axs = plt.subplots(1, 2, figsize=(TEXT_W * 0.62, 1.75))
    x = np.arange(1, 13)
    for run, lab, c, ls in (("viper", "ViPER", COLOR["viper"], "-"), ("abl_no_gates", "ViPER w/o gates", GREY, "--")):
        ks = [k for k in A if k.rsplit("_s", 1)[0] == run]
        if not ks:
            continue
        tex = np.array([A[k]["textured"] for k in ks]); smo = np.array([A[k]["smooth"] for k in ks])
        for ax, y in ((axs[0], None), (axs[1], tex - smo)):
            if y is None:
                ax.plot(x, tex.mean(0), color=c, ls=ls, label=f"{lab}, textured")
                ax.plot(x, smo.mean(0), color=c, ls=ls, alpha=0.45, label=f"{lab}, smooth")
            else:
                ax.errorbar(x, y.mean(0), yerr=y.std(0), color=c, ls=ls, marker="o", ms=2.5, capsize=1.5, label=lab)
    axs[0].set_ylabel("mean attention distance"); axs[0].set_xlabel("layer"); axs[0].legend(frameon=False, fontsize=6)
    axs[1].axhline(0, color=INK2, lw=0.5); axs[1].set_ylabel("textured $-$ smooth"); axs[1].set_xlabel("layer")
    axs[1].legend(frameon=False, fontsize=6)
    for ax in axs:
        ax.grid(True); ax.set_xticks([1, 4, 8, 12])
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_supp_attntex")
    return "ok"


def main():
    status = {"fig_curves": fig_curves(),
        "fig_curves_main": fig_curves_main(), "fig_resolution": fig_resolution_main(), "fig_curves_res": fig_curves_res(), "fig_supp_resolution": fig_resolution(), "fig_pilots": fig_pilots(), "fig_twopath": fig_twopath(), "fig_analysis": fig_analysis(),
              "fig_gatemaps": fig_gatemaps(), "fig_cost_pair": fig_cost_pair(),
              "fig_supp_paired": fig_supp_paired(), "fig_supp_attn": fig_supp_attn(), "fig_supp_gatebars": fig_supp_gatebars(), "fig_gatemaps_main": fig_gatemaps_main(),
              "fig_supp_gates": fig_supp_gates(), "fig_supp_probe": fig_supp_probe(), "fig_supp_robust": fig_supp_robust(),
              "fig_supp_texture": fig_supp_texture(), "fig_gate_pair": fig_gate_pair(),
              "fig_supp_gatectl": fig_supp_gatectl(), "fig_supp_kernel": fig_supp_kernel(),
              "fig_supp_attntex": fig_supp_attntex()}
    for k, v in status.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
