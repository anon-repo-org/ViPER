"""ADE20K semantic segmentation by fine-tuning an ImageNet-100 checkpoint (dense-prediction study).

python -m viper.seg --init results/in100_deit_small/viper_s0 --out results/ade20k_deit_small/viper_s0

Backbone: the ViT + PE of the classification run (same pe / pe_kwargs, weights from best.pt, head dropped),
run at 512x512 (32x32 tokens). Decoder: Segmenter's linear decoder (Strudel et al., ICCV'21; also used by
LookHere): LayerNorm'd patch tokens -> Linear(C -> 150) -> bilinear upsampling, so the comparison isolates
the backbone and its positional encoding. Recipe (mmseg ADE20K conventions, shortened schedule):
random resize 0.5-2.0 of (2048, 512), random 512 crop (cat_max_ratio 0.75), h-flip; AdamW, lr 1e-4
(head 10x), poly decay with linear warm-up, bf16. Eval: short side 512, sliding 512 window, stride 341,
single scale; mIoU over 150 classes (label 0 = ignore), plus aAcc.
Writes <out>/{config.json, history.json, last.pt, final.json, log.txt}; resumes from last.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from viper.models.vit import build_model

ROOT = Path(__file__).resolve().parents[1]
ADE = Path(os.environ.get("VIPER_DATA", ROOT / "data")) / "ade20k" / "ADEChallengeData2016"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
N_CLS, IGNORE = 150, 255


class ADE20K(Dataset):
    def __init__(self, split: str, crop: int = 512, train: bool = True):
        sub = "training" if split == "train" else "validation"
        self.imgs = sorted((ADE / "images" / sub).glob("*.jpg"))
        self.ann = ADE / "annotations" / sub
        self.crop, self.train = crop, train

    def __len__(self):
        return len(self.imgs)

    def load(self, i):
        img = Image.open(self.imgs[i]).convert("RGB")
        lab = Image.open(self.ann / (self.imgs[i].stem + ".png"))
        return img, lab

    @staticmethod
    def to_tensor(img, lab):
        x = (torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.).permute(2, 0, 1) - MEAN) / STD
        y = torch.from_numpy(np.asarray(lab, dtype=np.int64)) - 1          # reduce_zero_label: 0 -> -1
        y[y < 0] = IGNORE
        return x, y

    def __getitem__(self, i):
        img, lab = self.load(i)
        if not self.train:                                                 # short side -> crop, keep ratio
            w, h = img.size
            s = self.crop / min(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
            return self.to_tensor(img, lab)                                # label stays at full resolution
        w, h = img.size
        s = random.uniform(0.5, 2.0) * min(2048 / max(w, h), 512 / min(w, h))
        nw, nh = max(int(w * s + 0.5), 1), max(int(h * s + 0.5), 1)
        img, lab = img.resize((nw, nh), Image.BILINEAR), lab.resize((nw, nh), Image.NEAREST)
        x, y = self.to_tensor(img, lab)
        c = self.crop
        if nh < c or nw < c:                                               # pad to the crop size
            ph, pw = max(c - nh, 0), max(c - nw, 0)
            x = F.pad(x, (0, pw, 0, ph))
            y = F.pad(y, (0, pw, 0, ph), value=IGNORE)
        H, W = y.shape
        for _ in range(10):                                                # cat_max_ratio 0.75
            t, l = random.randint(0, H - c), random.randint(0, W - c)
            yc = y[t:t + c, l:l + c]
            v, n = torch.unique(yc[yc != IGNORE], return_counts=True)
            if len(n) > 1 and n.max() / n.sum() < 0.75:
                break
        x, y = x[:, t:t + c, l:l + c], y[t:t + c, l:l + c]
        if random.random() < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        return x, y


class SegViT(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        dim = vit.backbone.embed_dim
        self.head = nn.Linear(dim, N_CLS)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B, _, H, W = x.shape
        p = self.vit.patch
        t = self.vit.tokens(x)[:, self.vit.backbone.num_prefix_tokens:]
        logits = self.head(t).transpose(1, 2).reshape(B, N_CLS, H // p, W // p)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)


@torch.no_grad()
def slide_logits(model, x, crop=512, stride=341):
    """x: (1, 3, H, W) -> (1, N_CLS, H, W), sliding window with averaged overlaps."""
    _, _, H, W = x.shape
    out = torch.zeros(1, N_CLS, H, W, device=x.device)
    cnt = torch.zeros(1, 1, H, W, device=x.device)
    hs = list(range(0, max(H - crop, 0) + 1, stride))
    ws = list(range(0, max(W - crop, 0) + 1, stride))
    if hs[-1] + crop < H:
        hs.append(H - crop)
    if ws[-1] + crop < W:
        ws.append(W - crop)
    for t in hs:
        for l in ws:
            patch = x[:, :, t:t + crop, l:l + crop]
            ph, pw = patch.shape[-2:]
            if ph < crop or pw < crop:
                patch = F.pad(patch, (0, crop - pw, 0, crop - ph))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(patch).float()
            out[:, :, t:t + ph, l:l + pw] += lg[:, :, :ph, :pw]
            cnt[:, :, t:t + ph, l:l + pw] += 1
    return out / cnt


@torch.no_grad()
def evaluate(model, ds, device, limit=None):
    model.eval()
    inter = torch.zeros(N_CLS, dtype=torch.float64)
    union = torch.zeros(N_CLS, dtype=torch.float64)
    correct = total = 0
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        x, y = ds[i]
        lg = slide_logits(model, x[None].to(device))
        lg = F.interpolate(lg, size=y.shape, mode="bilinear", align_corners=False)
        pred = lg.argmax(1)[0].cpu()
        m = y != IGNORE
        p, t = pred[m], y[m]
        correct += (p == t).sum().item(); total += m.sum().item()
        inter += torch.bincount(t[p == t], minlength=N_CLS).double()
        area_p = torch.bincount(p, minlength=N_CLS).double()
        area_t = torch.bincount(t, minlength=N_CLS).double()
        union += area_p + area_t - torch.bincount(t[p == t], minlength=N_CLS).double()
    iou = inter / union.clamp(min=1)
    valid = union > 0
    model.train()
    return dict(miou=100 * iou[valid].mean().item(), aacc=100 * correct / max(total, 1),
                iou=[round(100 * v, 2) for v in iou.tolist()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="classification run dir (config.json + best.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=16, help="effective batch size per optimiser step")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (same effective batch, less memory)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr_mult", type=float, default=10.0)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_limit", type=int, default=None, help="debug: evaluate on the first N images")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "log.txt", "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True); fh.write(line + "\n"); fh.flush()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    init = Path(args.init).resolve()
    icfg = json.load(open(init / "config.json"))
    vit = build_model(backbone=icfg["backbone"], pe=icfg["pe"], num_classes=icfg["n_cls"],
                      img_size=icfg["img_size"], pretrained=False, drop_path=icfg["drop_path"],
                      pe_kwargs=json.loads(icfg["pe_kwargs"]))
    vit.load_state_dict(torch.load(init / "best.pt", map_location="cpu")["model"])
    model = SegViT(vit).to(device)

    import subprocess
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    cfg = dict(vars(args), init_resolved=str(init), init_config=icfg, git_commit=git, gpu=torch.cuda.get_device_name())
    json.dump(cfg, open(out / "config.json", "w"), indent=2)

    head = [p for n, p in model.named_parameters() if n.startswith("head.")]
    decay, no_decay = [], []
    for n, p in model.vit.named_parameters():
        if n.startswith("backbone.head"):                                   # classification head is unused
            p.requires_grad_(False); continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd, "lr_mult": 1.0},
                             {"params": no_decay, "weight_decay": 0.0, "lr_mult": 1.0},
                             {"params": head, "weight_decay": args.wd, "lr_mult": args.head_lr_mult}],
                            lr=args.lr, betas=(0.9, 0.999))

    it0, history = 0, []
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        it0, history = ck["iter"], ck["history"]
        log(f"resumed at iter {it0}")

    tr = ADE20K("train", args.crop, train=True)
    va = ADE20K("val", args.crop, train=False)
    g = torch.Generator(); g.manual_seed(args.seed + it0)
    assert args.bs % args.accum == 0
    dl = DataLoader(tr, batch_size=args.bs // args.accum, shuffle=True, num_workers=args.workers, pin_memory=True,
                    drop_last=True, generator=g, persistent_workers=True)
    log(f"init {init.name} pe={icfg['pe']} kw={icfg['pe_kwargs']}; train {len(tr)} val {len(va)}; iters {args.iters}")

    def lr_at(i):
        if i < args.warmup:
            return args.lr * (i + 1) / args.warmup
        return args.lr * (1 - (i - args.warmup) / max(args.iters - args.warmup, 1)) ** 0.9

    model.train()
    it, t0, loss_acc, micro = it0, time.time(), 0.0, 0
    while it < args.iters:
        for x, y in dl:
            if it >= args.iters:
                break
            lr = lr_at(it)
            for gr in opt.param_groups:
                gr["lr"] = lr * gr["lr_mult"]
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x).float(), y, ignore_index=IGNORE)
            (loss / args.accum).backward()
            micro += 1
            if micro < args.accum:
                continue
            micro = 0
            opt.step()
            opt.zero_grad(set_to_none=True)
            loss_acc += loss.item(); it += 1
            if it % 50 == 0:
                history.append(dict(iter=it, loss=loss_acc / 50, lr=lr, s_per_it=(time.time() - t0) / 50))
                log(f"iter {it} loss {loss_acc / 50:.4f} lr {lr:.2e} {(time.time() - t0) / 50:.3f}s/it")
                loss_acc, t0 = 0.0, time.time()
            if it % 1000 == 0 or it == args.iters:
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), iter=it, history=history), out / "last.pt")

    res = evaluate(model, va, device, args.eval_limit)
    log(f"FINAL seg_miou={res['miou']:.2f} aAcc={res['aacc']:.2f}")
    torch.save(dict(model=model.state_dict(), iter=it), out / "final.pt")
    final = dict(val=res, iters=it, pe=icfg["pe"], pe_kwargs=icfg["pe_kwargs"], init=str(init), config=cfg,
                 train_time_h=sum(h["s_per_it"] * 50 for h in history) / 3600, history=history)
    json.dump(final, open(out / "final.json", "w"), indent=2)
    (out / "last.pt").unlink(missing_ok=True)
    log("done")


if __name__ == "__main__":
    main()
