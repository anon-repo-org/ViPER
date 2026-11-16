#!/usr/bin/env python
"""Analysis of trained ImageNet-100 models (no training). Writes results/analysis/*.json and paper figures.

python -m scripts.analysis gates   # ViPER gate statistics + gate maps          -> fig_gates
python -m scripts.analysis attn    # mean attention distance per layer          -> fig_attndist

gates: for the ViPER checkpoints (seeds 0-2) on the first --n test images: mean gate per (layer, band), averaged
over heads and tokens; its spatial spread (std over tokens); and the Spearman correlation, per layer and band,
between a token's gate (mean over heads) and its local log wavelet energy at the finest level (sum of the three
orientations), i.e. does a textured patch weight the fine bands differently from a smooth one.
attn: mean attention distance (in patches) from every patch query to all patch keys, weighted by attention,
per layer, averaged over heads and images, for ViPER, its matched control and the strongest baselines.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from viper.data.datasets import get_dataset  # noqa: E402
from viper.models.vit import build_model  # noqa: E402

RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "analysis")


def load(run):
    d = os.path.join(RES, run)
    cfg = json.load(open(os.path.join(d, "config.json")))
    m = build_model(backbone=cfg["backbone"], pe=cfg["pe"], num_classes=cfg["n_cls"], img_size=cfg["img_size"],
                    pretrained=False, drop_path=0.0, pe_kwargs=json.loads(cfg["pe_kwargs"]))
    m.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location="cpu")["model"])
    return m.cuda().eval()


def loader(n, bs=64):
    (_, _, te, _), _ = get_dataset("imagenet100", img_size=224, aug="deit")
    return DataLoader(Subset(te, range(n)), batch_size=bs, shuffle=False, num_workers=6)


def spearman(a, b):
    ra, rb = a.argsort(-1).argsort(-1).float(), b.argsort(-1).argsort(-1).float()
    ra, rb = ra - ra.mean(-1, keepdim=True), rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp(min=1e-8)


@torch.no_grad()
def gates(n=2000, seeds=(0, 1, 2, 3, 4)):
    out = {}
    for s in seeds:
        m = load(f"in100_deit_small/viper_s{s}")
        pe = m.pe
        sums, sq, cnt, rho = 0, 0, 0, []
        for x, _ in loader(n):
            x = x.cuda()
            g = pe.gates(x, (14, 14))[..., :pe.n_bands].float()                # (B, L, H, N, nb)
            sums = sums + g.mean(2).mean(2).sum(0)                            # (L, nb): mean over heads, tokens
            sq = sq + g.mean(2).std(2).sum(0)                                 # spatial std over tokens
            cnt += x.shape[0]
            f = pe.energy_features(x, (14, 14))                               # (B, 3J+1, h, w), normalised log-energy
            fine = f[:, :3].mean(1).flatten(1)                                # finest level, mean over orientations
            gm = g.mean(2)                                                    # (B, L, N, nb)
            rho.append(spearman(gm.permute(0, 1, 3, 2), fine[:, None, None, :].expand_as(gm.permute(0, 1, 3, 2))).cpu())
        out[s] = dict(mean=(sums / cnt).cpu().tolist(), spatial_std=(sq / cnt).cpu().tolist(),
                      rho_fine=torch.cat(rho).mean(0).tolist())                # (L, nb)
        print("seed", s, "mean gate per band (avg over layers):", np.round(np.mean(out[s]["mean"], 0), 3),
              "rho(fine energy) per band:", np.round(np.mean(out[s]["rho_fine"], 0), 3), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "gates.json"), "w"), indent=1)
    # gate maps of seed 0 for a few images
    m = load("in100_deit_small/viper_s0")
    (_, _, te, _), _ = get_dataset("imagenet100", img_size=224, aug="deit")
    idx = [3, 517, 1210, 2750]
    x = torch.stack([te[i][0] for i in idx]).cuda()
    g = m.pe.gates(x, (14, 14))[..., :m.pe.n_bands].float().mean(2)         # (B, L, N, nb)
    f = m.pe.energy_features(x, (14, 14))[:, :3].mean(1)
    np.savez(os.path.join(OUT, "gate_maps.npz"), img=x.cpu().numpy(), gates=g.cpu().numpy(), fine=f.cpu().numpy(), idx=idx)
    return "ok"


@torch.no_grad()
def attn(n=512, runs=("viper", "abl_no_gates", "rope_mixed", "string_cayley", "lepe", "lookhere", "carope"), seeds=(0,)):
    ys, xs = torch.meshgrid(torch.arange(14.), torch.arange(14.), indexing="ij")
    p = torch.stack([ys.flatten(), xs.flatten()], -1).cuda()
    dist = torch.cdist(p, p)                                                  # (196, 196) in patches
    out = {}
    for r in runs:
        for s in seeds:
            d = os.path.join(RES, "in100_deit_small", f"{r}_s{s}")
            if not os.path.exists(os.path.join(d, "best.pt")):
                print("skip", r, s); continue
            m = load(f"in100_deit_small/{r}_s{s}")
            clear = m.ctx.clear

            def keep_clear():
                clear(); m.ctx.extra["keep_attn"] = True
            m.ctx.clear = keep_clear
            acc, cnt = torch.zeros(len(m.backbone.blocks), device="cuda"), 0
            for x, _ in loader(n, bs=32):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    m(x.cuda())
                for li, blk in enumerate(m.backbone.blocks):
                    a = blk.attn.last_attn.float()[:, :, 1:, 1:]                # patch -> patch
                    a = a / a.sum(-1, keepdim=True)
                    acc[li] += (a * dist).sum(-1).mean((1, 2)).sum()
                cnt += x.shape[0]
            out[f"{r}_s{s}"] = (acc / cnt).cpu().tolist()
            print(r, s, np.round(out[f"{r}_s{s}"], 2), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "attn_distance.json"), "w"), indent=1)
    return "ok"


def _fine(pe, x, grid=(14, 14)):
    return pe.energy_features(x, grid)[:, :3].mean(1).flatten(1)            # (B, N) normalised fine log-energy


@torch.no_grad()
def equiv(n=256, seed=0):
    """Shift test for Prop. 2 with the per-image GroupNorm included. Shift the image by one patch (16 px) and compare
    the gates of interior tokens (>= 4 patches from every border) with the gates of the same content before the
    shift. circular: torch.roll (image statistics unchanged); crop: reflect-pad + crop (content enters/leaves)."""
    m = load(f"in100_deit_small/viper_s{seed}")
    pe, P, G = m.pe, 16, 14
    res = {"circular": [], "crop": [], "spread": []}
    inner = [(r, c) for r in range(5, G - 4) for c in range(5, G - 4)]       # targets; sources (r-1, c-1) also >= 4 in
    tgt = torch.tensor([r * G + c for r, c in inner]).cuda()
    src = torch.tensor([(r - 1) * G + (c - 1) for r, c in inner]).cuda()
    for x, _ in loader(n):
        x = x.cuda()
        g0 = pe.gates(x, (G, G))[..., :pe.n_bands].float()                     # (B, L, H, N, nb)
        xr = torch.roll(x, (P, P), (2, 3))
        xc = torch.nn.functional.pad(x, (P, 0, P, 0), mode="reflect")[..., :x.shape[2], :x.shape[3]]
        for k, xs in (("circular", xr), ("crop", xc)):
            g1 = pe.gates(xs, (G, G))[..., :pe.n_bands].float()
            res[k].append((g1[..., tgt, :] - g0[..., src, :]).abs().flatten(1).cpu())
        res["spread"].append(g0[..., src, :].std(3).flatten(1).cpu())          # spread of gates across tokens
    out = {k: dict(mean=float(torch.cat(v).mean()), max=float(torch.cat(v).max()),
                   p99=float(torch.quantile(torch.cat(v).flatten()[:2_000_000], 0.99))) for k, v in res.items()}
    print(json.dumps(out, indent=1))
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "equiv.json"), "w"), indent=1)
    return "ok"


@torch.no_grad()
def kernel(seed=0, R=13):
    """Effective positional kernel of a query: k_i(D) = mean_h sum_b g_{l,h,b}(i) K_{l,h,b}(D), with the
    content-free band kernels K_b(D) = mean_{m in band b} cos<w_m, D> of the learned frequencies. Compared for the
    most textured and the smoothest interior query of a few test images, and for g = 1 (the matched control)."""
    m = load(f"in100_deit_small/viper_s{seed}")
    pe = m.pe
    (_, _, te, _), _ = get_dataset("imagenet100", img_size=224, aug="deit")
    idx = [3, 517, 1210, 2750]
    x = torch.stack([te[i][0] for i in idx]).cuda()
    g = pe.gates(x, (14, 14))[..., :pe.n_bands].float()                        # (B, L, H, N, nb)
    fine = _fine(pe, x)
    f = pe.rot.freqs.float()                                                   # (L, H, 2, P): rows (fx, fy)
    L, H, _, Pn = f.shape
    nb, na = pe.n_bands, Pn // 2
    band = torch.tensor([(p % na) // (na // nb) for p in range(Pn)]).cuda()    # band of every pair
    dy, dx = torch.meshgrid(torch.arange(-R, R + 1.), torch.arange(-R, R + 1.), indexing="ij")
    D = torch.stack([dx.flatten(), dy.flatten()], -1).cuda()                   # (M, 2) as (x, y)
    ang = torch.einsum("md,lhdp->lhmp", D, f)                                  # (L, H, M, P)
    cosv = ang.cos()
    Kb = torch.stack([cosv[..., band == b].mean(-1) for b in range(nb)], -1)   # (L, H, M, nb)
    inner = torch.tensor([r * 14 + c for r in range(3, 11) for c in range(3, 11)]).cuda()
    out = {"R": R, "idx": idx, "layers": {}}
    spread = g[:, :, :, inner].std(3).mean((0, 2, 3))                          # (L,) gate spread over tokens
    out["spread"] = spread.cpu().tolist()
    for l in range(L):
        k1 = Kb[l].sum(-1).mean(0)                                             # g = 1
        ks = {}
        for name, pick in (("textured", fine[:, inner].argmax(1)), ("smooth", fine[:, inner].argmin(1))):
            qi = inner[pick]                                                   # (B,)
            gi = g[:, l][torch.arange(len(idx)), :, qi]                         # (B, H, nb)
            ks[name] = torch.einsum("hmb,ahb->am", Kb[l], gi).div(H).mean(0)   # mean over images
            ks[name + "_g"] = gi.mean((0, 1)).cpu().tolist()
        out["layers"][l] = dict(ones=k1.view(2 * R + 1, -1).cpu().tolist(),
                                textured=ks["textured"].view(2 * R + 1, -1).cpu().tolist(),
                                smooth=ks["smooth"].view(2 * R + 1, -1).cpu().tolist(),
                                g_textured=ks["textured_g"], g_smooth=ks["smooth_g"])
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "kernel.json"), "w"))
    print("gate spread per layer:", np.round(out["spread"], 3))
    return "ok"


@torch.no_grad()
def attn_tex(n=512, seeds=(0, 1, 2), runs=("viper", "abl_no_gates")):
    """Mean attention distance of the most textured vs the smoothest fifth of query patches (by fine wavelet energy
    of the image), per layer, for ViPER and its matched control: do the gates change where queries attend?"""
    ys, xs = torch.meshgrid(torch.arange(14.), torch.arange(14.), indexing="ij")
    p = torch.stack([ys.flatten(), xs.flatten()], -1).cuda()
    dist = torch.cdist(p, p)
    ref = load("in100_deit_small/viper_s0").pe                                  # fixed texture measure for all runs
    out = {}
    for r in runs:
        for s in seeds:
            m = load(f"in100_deit_small/{r}_s{s}")
            clear = m.ctx.clear

            def keep_clear():
                clear(); m.ctx.extra["keep_attn"] = True
            m.ctx.clear = keep_clear
            acc = {"textured": torch.zeros(12, device="cuda"), "smooth": torch.zeros(12, device="cuda")}
            cnt = 0
            for x, _ in loader(n, bs=32):
                x = x.cuda()
                fine = _fine(ref, x)                                           # (B, N)
                q = fine.quantile(torch.tensor([0.2, 0.8], device="cuda"), dim=1)   # (2, B)
                masks = {"smooth": fine <= q[0][:, None], "textured": fine >= q[1][:, None]}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    m(x)
                for li, blk in enumerate(m.backbone.blocks):
                    a = blk.attn.last_attn.float()[:, :, 1:, 1:]
                    a = a / a.sum(-1, keepdim=True)
                    d = (a * dist).sum(-1).mean(1)                              # (B, N) mean over heads
                    for k, mk in masks.items():
                        acc[k][li] += ((d * mk).sum(1) / mk.sum(1)).sum()
                cnt += x.shape[0]
            out[f"{r}_s{s}"] = {k: (v / cnt).cpu().tolist() for k, v in acc.items()}
            print(r, s, {k: np.round(v, 2)[[0, 3, 6, 9, 11]] for k, v in out[f"{r}_s{s}"].items()}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "attn_texture.json"), "w"), indent=1)
    return "ok"


@torch.no_grad()
def gatecost(bs=128, G=14):
    """Cost of ViPER's wavelet branch (DWT + gate network) on DeiT-S at 224^2: multiply-adds, parameters and its share
    of the forward time (bf16, one GPU)."""
    import time
    m = build_model(backbone="deit_small", pe="viper", num_classes=100, img_size=224, pretrained=False, drop_path=0.0,
                    pe_kwargs={"lepe": True}).cuda().eval()
    pe, Lf, h, dwt = m.pe, 8, 224, 0
    for _ in range(pe.levels):
        h //= 2
        dwt += 4 * Lf * Lf * h * h                                              # 4 separable-product filters per level
    nf, hid = 3 * pe.levels + 1, pe.hidden[0].out_channels
    nout = pe.head.out_channels
    net = (nf * 9 + nf * hid + hid * nout) * G * G
    params = sum(p.numel() for n, p in pe.named_parameters() if not n.startswith("rot"))
    x = torch.randn(bs, 3, 224, 224, device="cuda")

    def timeit(f, n):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                f()
            torch.cuda.synchronize(); t = time.time()
            for _ in range(n):
                f()
            torch.cuda.synchronize()
        return (time.time() - t) / n
    tg, tm = timeit(lambda: pe.gates(x, (G, G)), 30), timeit(lambda: m(x), 10)
    out = dict(dwt_mmac=dwt / 1e6, net_mmac=net / 1e6, total_mmac=(dwt + net) / 1e6, params=params, time_share=tg / tm)
    print(out)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "gate_cost.json"), "w"), indent=1)
    return "ok"


@torch.no_grad()
def gatefreeze(seeds=(0, 1, 2, 3, 4), n_val=2000):
    """Test-time gate interventions on trained ViPER (ImageNet-100 test set, 224^2). Each variant replaces the gates
    g (B, L, H, N, nb) before attention: full (unchanged), img_mean (each image's mean over tokens: keeps the
    per-image level, removes per-token variation), shuffle (tokens permuted within the image: keeps the per-image
    distribution, breaks the alignment with the image), data_mean (mean over 2,000 validation-holdout images per
    layer, head and band: removes all image dependence), ones (g = 1: the trained model without its gates)."""
    (_, va, te, _), _ = get_dataset("imagenet100", img_size=224, aug="deit")
    vl = DataLoader(Subset(va, range(n_val)), batch_size=64, shuffle=False, num_workers=6)
    tl = DataLoader(te, batch_size=64, shuffle=False, num_workers=6)
    out = {}
    for s in seeds:
        m = load(f"in100_deit_small/viper_s{s}")
        pe = m.pe
        orig = pe.gates
        acc, cnt = 0, 0
        for x, _ in vl:                                                     # dataset-mean gates from the val holdout
            g = orig(x.cuda(), (14, 14)).float()
            acc, cnt = acc + g.mean(3).sum(0), cnt + x.shape[0]
        gmean = acc / cnt                                                   # (L, H, nb)
        mode = {"v": "full"}

        def gates(image, grid):
            g = orig(image, grid)
            v = mode["v"]
            if v == "img_mean":
                return g.mean(3, keepdim=True).expand_as(g)
            if v == "shuffle":
                perm = torch.randperm(g.shape[3], device=g.device)
                return g[:, :, :, perm]
            if v == "data_mean":
                return gmean[None, :, :, None, :].to(g.dtype).expand_as(g)
            if v == "ones":
                return torch.ones_like(g)
            return g
        pe.gates = gates
        res = {}
        for v in ("full", "img_mean", "shuffle", "data_mean", "ones"):
            mode["v"] = v
            torch.manual_seed(0)
            corr, tot = 0, 0
            for x, y in tl:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    corr += (m(x.cuda()).argmax(-1).cpu() == y).sum().item()
                tot += len(y)
            res[v] = 100 * corr / tot
        out[s] = res
        print("seed", s, {k: round(v, 2) for k, v in res.items()}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "gatefreeze.json"), "w"), indent=1)
    return "ok"


@torch.no_grad()
def gatevar(n=2000, seeds=(0, 1, 2, 3, 4)):
    """Variance of the gates split into between-image (variance of each image's mean gate) and within-image
    (mean over images of the variance across tokens), per layer, head and band, averaged; the last layer is excluded
    (its gates get no gradient)."""
    out = {}
    for s in seeds:
        pe = load(f"in100_deit_small/viper_s{s}").pe
        means, wvar = [], []
        for x, _ in loader(n):
            g = pe.gates(x.cuda(), (14, 14))[..., :pe.n_bands].float()[:, :-1]   # (B, L-1, H, N, nb)
            means.append(g.mean(3).cpu()); wvar.append(g.var(3).cpu())
        mu, wv = torch.cat(means), torch.cat(wvar)                               # (I, L-1, H, nb)
        between, within = mu.var(0).mean().item(), wv.mean().item()
        out[s] = dict(between=between, within=within, frac_within=within / (between + within))
        print("seed", s, {k: round(v, 4) for k, v in out[s].items()}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "gatevar.json"), "w"), indent=1)
    return "ok"


def bandorder(seeds=(0, 1, 2, 3, 4)):
    """Do the learned RoPE-Mixed frequencies of ViPER keep the fine -> coarse band order? Per layer and head: mean
    frequency norm of every band; fraction of (layer, head) with strictly decreasing band means, and fraction of pairs
    whose frequency norm lies inside the range spanned by its own band's initial frequencies (+-25%)."""
    import math
    out = {}
    for s in seeds:
        sd = torch.load(os.path.join(RES, f"in100_deit_small/viper_s{s}", "best.pt"), map_location="cpu")["model"]
        f = [v for k, v in sd.items() if k.endswith("rot.freqs")][0].float()   # (L, H, 2, P)
        L, H, _, P = f.shape
        na, nb = P // 2, 4
        nrm = f.norm(dim=2)                                                     # (L, H, P)
        band = torch.tensor([(p % na) // (na // nb) for p in range(P)])
        bm = torch.stack([nrm[..., band == b].mean(-1) for b in range(nb)], -1)  # (L, H, nb)
        ordered = (bm[..., :-1] > bm[..., 1:]).all(-1).float().mean().item()
        init = torch.tensor([100.0 ** (-(m % na) / na) for m in range(P)])     # theta=100 initial frequencies
        lo = torch.stack([init[band == b].min() for b in range(nb)]) * 0.75
        hi = torch.stack([init[band == b].max() for b in range(nb)]) * 1.25
        inside = ((nrm >= lo[band]) & (nrm <= hi[band])).float().mean().item()
        out[s] = dict(ordered=ordered, inside=inside, band_means=bm.mean((0, 1)).tolist())
        print("seed", s, round(ordered, 3), round(inside, 3), [round(v, 3) for v in out[s]["band_means"]], flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "bandorder.json"), "w"), indent=1)
    return "ok"


if __name__ == "__main__":
    {"gates": gates, "attn": attn, "equiv": equiv, "kernel": kernel, "attn_tex": attn_tex, "gatecost": gatecost, "gatefreeze": gatefreeze, "gatevar": gatevar, "bandorder": bandorder}[sys.argv[1]]()
