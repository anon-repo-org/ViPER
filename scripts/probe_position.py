#!/usr/bin/env python
"""Coordinate probing: how much *position* do the tokens of a trained model carry?

For every block output we fit a ridge regression from the (centred) token features to
the patch coordinates (y, x) on held-out images and report R^2 on other images. A PE
that injects no position gives R^2 ≈ 0 at layer 0 (tokens are content only).

python scripts/probe_position.py results/pilot_in100_30ep/viper_s0 --n_img 512
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from viper.data.datasets import get_dataset
from viper.models.vit import build_model


@torch.no_grad()
def collect(model, loader, device, layers):
    feats = {l: [] for l in layers}
    coords = []
    m = model.backbone
    for x, _ in loader:
        x = x.to(device)
        B = x.shape[0]
        grid = (x.shape[-2] // model.patch, x.shape[-1] // model.patch)
        # replicate ViTWithPE.tokens but keep intermediate outputs
        ctx = model.ctx
        ctx.clear(); ctx.grid = grid; ctx.extra["image"] = x; ctx.extra["pos_scale"] = 1.0
        h = m.patch_embed(x).reshape(B, -1, m.embed_dim)
        h = model.pe(h, ctx)
        h = torch.cat([m.cls_token.expand(B, -1, -1), h], 1)
        if 0 in feats:
            feats[0].append(h[:, 1:].float().cpu())
        for i, blk in enumerate(m.blocks):
            h = blk(h)
            if model.pe.kind == "conv":
                h = model.pe.after_block(i, h, ctx)
            if i + 1 in feats:
                feats[i + 1].append(h[:, 1:].float().cpu())
        gy, gx = torch.meshgrid(torch.arange(grid[0]), torch.arange(grid[1]), indexing="ij")
        c = torch.stack([gy.reshape(-1), gx.reshape(-1)], -1).float()
        coords.append(c[None].expand(B, -1, -1))
    return {l: torch.cat(v) for l, v in feats.items()}, torch.cat(coords)


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    A = Xtr.T @ Xtr + lam * Xtr.shape[0] * torch.eye(Xtr.shape[1])
    W = torch.linalg.solve(A, Xtr.T @ (Ytr - ym))
    pred = Xte @ W + ym
    ss_res = ((Yte - pred) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0)
    return (1 - ss_res / ss_tot).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n_img", type=int, default=512)
    ap.add_argument("--layers", default="0,1,2,4,6,8,10,12")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    device = "cuda"
    (tr, va, te, n_cls), dkw = get_dataset(cfg["dataset"], img_size=cfg["img_size"], aug="none")
    model = build_model(backbone=cfg["backbone"], pe=cfg["pe"], num_classes=n_cls, img_size=cfg["img_size"],
                        pe_kwargs=json.loads(cfg["pe_kwargs"])).to(device).eval()
    model.load_state_dict(torch.load(os.path.join(args.run, "best.pt"), map_location="cpu")["model"])
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(te), generator=g)[: args.n_img].tolist()
    loader = DataLoader(Subset(te, idx), batch_size=64, num_workers=4)
    layers = [int(v) for v in args.layers.split(",")]
    feats, coords = collect(model, loader, device, layers)
    n = coords.shape[0]; half = n // 2
    out = {}
    for l in layers:
        X = feats[l]; Y = coords
        Xtr, Ytr = X[:half].reshape(-1, X.shape[-1]), Y[:half].reshape(-1, 2)
        Xte, Yte = X[half:].reshape(-1, X.shape[-1]), Y[half:].reshape(-1, 2)
        r2 = ridge_r2(Xtr, Ytr, Xte, Yte)
        out[l] = {"r2_y": r2[0], "r2_x": r2[1]}
        print(f"layer {l:2d}: R^2 y={r2[0]:.3f} x={r2[1]:.3f}")
    json.dump(out, open(args.out or os.path.join(args.run, "probe_position.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
