#!/usr/bin/env python
"""Image assets for the architecture figure (paper/figures/arch_*.png), computed with the trained ViPER model.

python -m scripts.make_arch_assets [--run results/in100_deit_small/viper_s0]

Input: "Chelsea" from scikit-image (CC0, no copyright restrictions), centre-cropped and resized to 224x224.
Writes: arch_input.png (the input), arch_dwt.png (2-level Haar decomposition of the grey image, standard
pyramid layout, detail sub-bands contrast-stretched), arch_energy.png (fine-scale log wavelet energy pooled to
the 14x14 token grid, as fed to the gate network), arch_gate_b{1..4}.png (the gate of each frequency band for
this image, mean over layers and heads, one colour scale per band).
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


def haar2(x):
    a, b, c, d = x[0::2, 0::2], x[0::2, 1::2], x[1::2, 0::2], x[1::2, 1::2]
    return (a + b + c + d) / 2, (a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2


def save_map(arr, path, cmap, vmin=None, vmax=None):
    fig = plt.figure(figsize=(1, 1), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.savefig(path, dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(ROOT, "results", "in100_deit_small", "viper_s0"))
    args = ap.parse_args()
    from skimage import data
    img = data.chelsea()                                                      # (300, 451, 3) uint8, CC0
    h, w, _ = img.shape
    s = min(h, w); t, l = (h - s) // 2, (w - s) // 2 + 30                   # keep the face centred
    pil = Image.fromarray(img[t:t + s, l:l + s]).resize((224, 224), Image.BICUBIC)
    pil.save(os.path.join(OUT, "arch_input.png"))

    g = np.asarray(pil.convert("L"), dtype=np.float32) / 255.0
    ll1, lh1, hl1, hh1 = haar2(g)
    ll2, lh2, hl2, hh2 = haar2(ll1)
    st = lambda d: np.clip(np.abs(d) / (np.percentile(np.abs(d), 99) + 1e-6), 0, 1)
    top = np.block([[ll2 / ll2.max(), st(lh2)], [st(hl2), st(hh2)]])
    mosaic = np.block([[top, st(lh1)], [st(hl1), st(hh1)]])
    mosaic[111:113, :] = mosaic[:, 111:113] = 1.0                             # sub-band separators
    mosaic[55:57, :112] = mosaic[:112, 55:57] = 1.0
    save_map(mosaic, os.path.join(OUT, "arch_dwt.png"), "gray", 0, 1)

    cfg = json.load(open(os.path.join(args.run, "config.json")))
    m = build_model(backbone=cfg["backbone"], pe=cfg["pe"], num_classes=cfg["n_cls"], img_size=224,
                    pretrained=False, drop_path=0.0, pe_kwargs=json.loads(cfg["pe_kwargs"])).eval()
    m.load_state_dict(torch.load(os.path.join(args.run, "best.pt"), map_location="cpu")["model"])
    mean, std = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = ((torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.).permute(2, 0, 1) - mean) / std)[None]
    with torch.no_grad():
        f = m.pe.energy_features(x, (14, 14))[0, :3].mean(0).numpy()          # finest level, mean over orientations
        gate = m.pe.gates(x, (14, 14))[0, ..., :m.pe.n_bands].mean((0, 1)).numpy()  # (N, nb)
    save_map(f, os.path.join(OUT, "arch_energy.png"), "magma")
    for b in range(gate.shape[1]):                                           # own colour scale per band
        save_map(gate[:, b].reshape(14, 14), os.path.join(OUT, f"arch_gate_b{b + 1}.png"), "viridis")
    print("band means", np.round(gate.mean(0), 3), "band ranges", np.round(gate.min(0), 2), np.round(gate.max(0), 2))


if __name__ == "__main__":
    main()
