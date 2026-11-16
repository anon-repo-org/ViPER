#!/usr/bin/env python
"""Qualitative ADE20K examples for the supplementary material (paper/figures/fig_supp_seg.{pdf,png}).

python -m scripts.seg_examples [--seed 0] [--idx 0 ...]

Loads the fine-tuned segmentation models (results/ade20k_deit_small/<method>_s<seed>/final.pt), predicts the
chosen validation images with the same sliding-window inference as the evaluation, and shows image, ground truth
and predictions with a fixed 150-colour palette (ignored pixels black). Images are picked by index, not by
result, and the per-image mIoU of every model is printed in each panel.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "pdf.fonttype": 42})
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from viper.models.vit import build_model  # noqa: E402
from viper.seg import ADE20K, IGNORE, MEAN, N_CLS, STD, SegViT, slide_logits  # noqa: E402

METHODS = [("rope_mixed", "RoPE-Mixed"), ("string_cayley", "STRING"), ("lookhere", "LookHere"),
           ("abl_no_gates", "ViPER w/o gates"), ("viper", "ViPER (ours)")]


def palette():
    rng = np.random.RandomState(3)
    p = rng.randint(40, 255, size=(N_CLS, 3)).astype(np.uint8)
    return np.concatenate([p, np.zeros((1, 3), np.uint8)])                   # index N_CLS: ignore -> black


def colour(lab, pal):
    lab = lab.copy(); lab[lab == IGNORE] = N_CLS
    return pal[lab]


def miou(pred, y):
    m = y != IGNORE
    p, t = pred[m], y[m]
    ious = []
    for c in np.unique(t):
        i = ((p == c) & (t == c)).sum(); u = ((p == c) | (t == c)).sum()
        ious.append(i / u)
    return 100 * float(np.mean(ious))


def load(method, seed):
    d = os.path.join(ROOT, "results", "ade20k_deit_small", f"{method}_s{seed}")
    fin = json.load(open(os.path.join(d, "final.json")))
    icfg = fin["config"]["init_config"]
    vit = build_model(backbone=icfg["backbone"], pe=icfg["pe"], num_classes=icfg["n_cls"], img_size=icfg["img_size"],
                      pretrained=False, drop_path=0.0, pe_kwargs=json.loads(icfg["pe_kwargs"]))
    m = SegViT(vit)
    m.load_state_dict(torch.load(os.path.join(d, "final.pt"), map_location="cpu")["model"])
    return m.cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--idx", type=int, nargs="+", default=[3, 120, 640, 1500])
    args = ap.parse_args()
    ds = ADE20K("val", 512, train=False)
    pal = palette()
    methods = [(m, l) for m, l in METHODS
               if os.path.exists(os.path.join(ROOT, "results", "ade20k_deit_small", f"{m}_s{args.seed}", "final.pt"))]
    preds = {m: [] for m, _ in methods}
    for m, _ in methods:
        model = load(m, args.seed)
        for i in args.idx:
            x, y = ds[i]
            with torch.no_grad():
                lg = slide_logits(model, x[None].cuda())
                lg = F.interpolate(lg, size=y.shape, mode="bilinear", align_corners=False)
            preds[m].append(lg.argmax(1)[0].cpu().numpy())
        del model; torch.cuda.empty_cache()
    cols = 2 + len(methods)
    ratios = [ds[i][1].shape[0] / ds[i][1].shape[1] for i in args.idx]          # panel height / width per row
    W = 6.875; pw = W / cols
    fig, axes = plt.subplots(len(args.idx), cols, figsize=(W, pw * sum(ratios) + 0.2),
                             gridspec_kw=dict(height_ratios=ratios, wspace=0.03, hspace=0.04))
    for r, i in enumerate(args.idx):
        x, y = ds[i]
        img = np.clip((x * STD + MEAN).permute(1, 2, 0).numpy(), 0, 1)
        img = F.interpolate(torch.from_numpy(img).permute(2, 0, 1)[None], size=y.shape, mode="bilinear")[0].permute(1, 2, 0).numpy()
        y = y.numpy()
        panels = [(img, "image"), (colour(y, pal), "ground truth")]
        panels += [(colour(preds[m][r], pal), f"{lab}\n{miou(preds[m][r], y):.1f}") for m, lab in methods]
        for c, (pnl, t) in enumerate(panels):
            ax = axes[r, c]; ax.imshow(pnl, aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(t.split("\n")[0], fontsize=7, pad=2)
            if c >= 2:
                ax.text(0.03, 0.04, t.split("\n")[1], transform=ax.transAxes, fontsize=6, color="white",
                        bbox=dict(facecolor="black", alpha=0.55, pad=1, lw=0))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1 - 0.2 / (pw * sum(ratios) + 0.2))
    out = os.path.join(ROOT, "paper", "figures", "fig_supp_seg")
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    print("methods:", [m for m, _ in methods])


if __name__ == "__main__":
    main()
