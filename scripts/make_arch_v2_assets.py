#!/usr/bin/env python
"""Assets for the second version of the architecture figure (paper/figures/arch2_*), computed with the
trained ViPER model on the same CC0 input as make_arch_assets.py (run that first: it writes arch_input.png).

python -m scripts.make_arch_v2_assets [--run results/in100_deit_small/viper_s0] [--layer 3]

Writes arch2_spectrum.pdf (top: the four band kernels K_b(dx) of the default theta grid with unit
coefficients, i.e. the filter bank; bottom: the gates of one smooth and one textured patch of this image,
one layer, mean over heads) and arch2_tokens.json (grid positions of the two patches, marked in the figure).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from viper.models.vit import build_model  # noqa: E402

OUT = os.path.join(ROOT, "paper", "figures")
BAND = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd"]       # = bandA..bandD in the paper preamble
EDGE = ["#7a9cbf", "#5c8fb0", "#3f78a6", "#1d5a8a"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(ROOT, "results", "in100_deit_small", "viper_s0"))
    ap.add_argument("--layer", type=int, default=3)
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    m = build_model(backbone=cfg["backbone"], pe=cfg["pe"], num_classes=cfg["n_cls"], img_size=224,
                    pretrained=False, drop_path=0.0, pe_kwargs=json.loads(cfg["pe_kwargs"])).eval()
    m.load_state_dict(torch.load(os.path.join(args.run, "best.pt"), map_location="cpu")["model"])
    pil = Image.open(os.path.join(OUT, "arch_input.png")).convert("RGB")
    mean, std = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = ((torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.).permute(2, 0, 1) - mean) / std)[None]
    with torch.no_grad():
        e = m.pe.energy_features(x, (14, 14))[0, :3].mean(0).numpy().ravel()   # finest level, mean orientation
        g = m.pe.gates(x, (14, 14))[0, args.layer, :, :, :m.pe.n_bands].mean(0).numpy()  # (N, B), mean over heads
    rc = np.array([(i // 14, i % 14) for i in range(196)])
    inner = (rc.min(1) >= 2) & (rc.max(1) <= 11)                              # away from the border
    order = np.argsort(e)
    smooth = next(i for i in order if inner[i])
    tex = next(i for i in order[::-1] if inner[i])
    gs, gt = g[smooth], g[tex]
    json.dump({"smooth": [int(rc[smooth][0]), int(rc[smooth][1])], "textured": [int(rc[tex][0]), int(rc[tex][1])],
               "layer": args.layer, "g_smooth": gs.round(3).tolist(), "g_textured": gt.round(3).tolist()},
              open(os.path.join(OUT, "arch2_tokens.json"), "w"), indent=1)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 7.5, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
                         "ytick.major.size": 2, "pdf.fonttype": 42})
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(1.9, 2.95), gridspec_kw=dict(height_ratios=[1, 1.0], hspace=0.95))
    M, theta, B = 16, 100.0, 4
    f = theta ** (-np.arange(M) / M)
    d = np.linspace(-13, 13, 600)
    for b in range(B):
        k = np.cos(np.outer(d, f[b * 4:(b + 1) * 4])).mean(1)
        a0.plot(d, k, color=EDGE[b], lw=1.0, label=f"$K_{b + 1}$")
    a0.set_xlim(-13, 13); a0.set_ylim(-0.7, 1.1); a0.set_xticks([-12, -6, 0, 6, 12]); a0.set_yticks([0, 1])
    a0.set_xlabel(r"displacement $\Delta x$ (patches)", labelpad=1)
    a0.set_title("band kernels (fine $\\rightarrow$ coarse)", fontsize=7.5, pad=3)
    a0.legend(ncol=4, frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.36),
              handlelength=1.0, columnspacing=0.6, handletextpad=0.3)
    for s in ("top", "right"):
        a0.spines[s].set_visible(False); a1.spines[s].set_visible(False)
    xb = np.arange(B)
    a1.bar(xb - 0.19, gs, 0.36, color=BAND, edgecolor=EDGE, lw=0.6, label="smooth S")
    a1.bar(xb + 0.19, gt, 0.36, color=BAND, edgecolor="black", lw=0.6, hatch="////", label="textured T")
    a1.axhline(1.0, color="0.35", lw=0.6, ls=(0, (3, 2)))
    # dashed line = RoPE (g = 1); explained in the caption
    a1.set_xticks(xb); a1.set_xticklabels([f"$b={b + 1}$" for b in xb])
    a1.set_ylim(0, 2.05); a1.set_yticks([0, 1, 2]); a1.set_xlim(-0.65, 3.55)
    a1.set_ylabel(r"gain $g_b(i)$", labelpad=1)
    a1.set_title(f"spectral filter $G_i$ (layer {args.layer + 1})", fontsize=7.5, pad=3)
    leg = a1.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2, handlelength=1.2, handleheight=0.8,
                    borderaxespad=0.1, columnspacing=0.8, handletextpad=0.4)
    for h in leg.legend_handles:
        h.set_facecolor("white")
    fig.savefig(os.path.join(OUT, "arch2_spectrum.pdf"), bbox_inches="tight", pad_inches=0.01, transparent=True)
    print("smooth", rc[smooth], gs.round(2), "textured", rc[tex], gt.round(2))


if __name__ == "__main__":
    main()
