"""Trainer for PE comparisons (single GPU, or multi-GPU DDP under torchrun).

python -m viper.train --dataset imagenet100 --backbone deit_small --pe viper --seed 0 \
    --epochs 100 --out results/in100_deit_small/viper_s0
torchrun --nproc_per_node 2 -m viper.train --dataset imagenet1k --bs 170 --accum 3 ...   # global batch 1020

--bs is the per-GPU batch; the global batch is bs * GPUs * accum and the default lr scales with it.

Writes <out>/{config.json, history.json, last.pt, best.pt, final.json, log.txt}; a
line `FINAL test_top1=...` marks completion (the queue looks for it). Re-running
the same command auto-resumes from last.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from viper.data.datasets import get_dataset
from viper.models.vit import build_model

EVAL_RES = [64, 96, 128, 160, 224, 288, 320, 384, 448]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def log(fh, msg):
    if fh is None:                  # non-zero DDP rank
        return
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n"); fh.flush()


@torch.no_grad()
def evaluate(model, loader, device, pos_scale=1.0, amp=True, return_logits=False):
    model.eval()
    correct = top5 = n = 0
    loss_sum = 0.0
    preds, labels, logits = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(x, pos_scale=pos_scale)
        out = out.float()
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        k = min(5, out.shape[1])
        t5 = out.topk(k, dim=1).indices
        correct += (t5[:, 0] == y).sum().item()
        top5 += (t5 == y[:, None]).any(1).sum().item()
        n += y.numel()
        preds.append(t5[:, 0].cpu()); labels.append(y.cpu())
        if return_logits:
            logits.append(out.half().cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    from sklearn.metrics import balanced_accuracy_score, f1_score
    m = dict(top1=100 * correct / n, top5=100 * top5 / n, loss=loss_sum / n,
             macro_f1=100 * f1_score(labels, preds, average="macro"),
             bal_acc=100 * balanced_accuracy_score(labels, preds))
    if return_logits:
        return m, dict(labels=labels.astype(np.int16), logits=torch.cat(logits).numpy())
    return m


def cosine_lr(step, total, warmup, base, minimum):
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return minimum + 0.5 * (base - minimum) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="deit_small")
    ap.add_argument("--pe", default="viper")
    ap.add_argument("--pe_kwargs", default="{}", help="JSON dict forwarded to the PE constructor")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--init", default=None, help="transfer: initialise from this checkpoint (our own run), new head")
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--aug", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=128, help="per-GPU batch")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 5e-4 * B/512 (scratch) or 1e-4 * B/512 (pretrained), B = global batch")
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--gate_lr_mult", type=float, default=1.0,
                    help="ViPER: learning-rate multiplier for the gate branch (wavelet descriptor -> gates); "
                         "ImageNet-1k uses 0.125 so the gates keep the ImageNet-100 peak lr of 1.25e-4")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.8)
    ap.add_argument("--cutmix", type=float, default=1.0)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval_res", default=",".join(map(str, EVAL_RES)))
    ap.add_argument("--no_multires", action="store_true")
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop an epoch after n steps")
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank, local_rank = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        torch.cuda.set_device(local_rank)
    rank0 = rank == 0
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "log.txt", "a") if rank0 else None
    device = f"cuda:{local_rank}" if world > 1 else "cuda"
    set_seed(args.seed + rank)      # rank 0 = the single-GPU seed; DDP broadcasts rank 0's initial weights
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp = not args.no_amp

    (tr, va, te, n_cls), dkw = get_dataset(args.dataset, img_size=args.img_size, aug=args.aug)
    img_size = dkw["img_size"]
    if args.lr is None:
        args.lr = (1e-4 if (args.pretrained or args.init) else 5e-4) * args.bs * world * args.accum / 512
    import subprocess
    root = Path(__file__).resolve().parents[1]
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "viper"], capture_output=True,
                                text=True).stdout.strip())
    cfg = dict(vars(args)); cfg.update(img_size=img_size, aug=dkw["aug"], n_cls=n_cls,
                                      n_train=len(tr), n_val=len(va), n_test=len(te), world_size=world,
                                      global_bs=args.bs * world * args.accum,
                                      git_commit=git + ("-dirty" if dirty else ""),
                                      torch=torch.__version__, gpu=torch.cuda.get_device_name())
    if rank0:
        json.dump(cfg, open(out / "config.json", "w"), indent=2)
    log(fh, f"config: {json.dumps(cfg)}")

    g = torch.Generator(); g.manual_seed(args.seed)
    sampler = DistributedSampler(tr, num_replicas=world, rank=rank, shuffle=True, seed=args.seed,
                                 drop_last=True) if world > 1 else None
    tl = DataLoader(tr, batch_size=args.bs, shuffle=sampler is None, sampler=sampler, num_workers=args.workers,
                    pin_memory=True, drop_last=True, generator=g, persistent_workers=args.workers > 0)
    vl = DataLoader(va, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
    tel = DataLoader(te, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(backbone=args.backbone, pe=args.pe, num_classes=n_cls, img_size=img_size,
                        pretrained=args.pretrained, drop_path=args.drop_path,
                        pe_kwargs=json.loads(args.pe_kwargs)).to(device)
    if args.init and not (out / "last.pt").exists():
        sd = torch.load(args.init, map_location="cpu")["model"]
        own = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape and not (k.startswith("head.") or k.startswith("backbone.head."))}  # classifier only
        missing = model.load_state_dict(keep, strict=False).missing_keys
        log(fh, f"init from {args.init}: {len(keep)} tensors loaded, not loaded: {missing}")
    # some baselines (e.g. STRING) have parameters that never get a gradient: allow them under DDP
    net = (torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
           if world > 1 else model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(fh, f"model {args.backbone}+{args.pe}: {n_params/1e6:.2f}M params, PE params = {model.pe_parameters()}")

    GATE = ("pe.norm.", "pe.ctx_conv.", "pe.hidden.", "pe.head.")     # ViPER gate branch
    split_gate = args.gate_lr_mult != 1.0                               # default: same groups as before (resume-safe)
    decay, no_decay, g_decay, g_no_decay = [], [], [], []
    for n_, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nd = p.ndim <= 1 or n_.endswith(".bias") or "pe." in n_ and "table" in n_
        if split_gate and n_.startswith(GATE):
            (g_no_decay if nd else g_decay).append(p)
        else:
            (no_decay if nd else decay).append(p)
    groups = [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}]
    if split_gate:
        assert g_decay or g_no_decay, "--gate_lr_mult given but the model has no gate branch"
        groups += [{"params": g_decay, "weight_decay": args.wd, "lr_mult": args.gate_lr_mult},
                   {"params": g_no_decay, "weight_decay": 0.0, "lr_mult": args.gate_lr_mult}]
        log(fh, f"gate branch: {sum(p.numel() for p in g_decay + g_no_decay)} params at lr x {args.gate_lr_mult}")
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999))
    steps_per_epoch = len(tl) // args.accum
    total = args.epochs * steps_per_epoch
    warm = args.warmup * steps_per_epoch

    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        from timm.data import Mixup
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, label_smoothing=args.smoothing,
                         num_classes=n_cls)
        from timm.loss import SoftTargetCrossEntropy
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    start_ep, best_val, history = 0, -1.0, []
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_ep, best_val, history = ck["epoch"] + 1, ck["best_val"], ck["history"]
        log(fh, f"resumed from epoch {start_ep}")

    step = start_ep * steps_per_epoch
    for ep in range(start_ep, args.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(ep)
        t0, loss_sum, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        n_micro = steps_per_epoch * args.accum
        for i, (x, y) in enumerate(tl):
            if i >= n_micro or (args.max_steps and i >= args.max_steps * args.accum):
                break
            last_micro = (i + 1) % args.accum == 0
            if i % args.accum == 0:
                lr = cosine_lr(step, total, warm, args.lr, args.min_lr)
                for pg in opt.param_groups:
                    pg["lr"] = lr * pg.get("lr_mult", 1.0)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if mixup_fn is not None:
                x, y = mixup_fn(x, y)
            with (net.no_sync() if world > 1 and not last_micro else nullcontext()):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    out_ = net(x)
                    loss = criterion(out_, y)
                (loss / args.accum).backward()
            loss_sum += loss.item(); nb += 1
            if not math.isfinite(loss.item()):
                log(fh, "non-finite loss, aborting") if rank0 else print("non-finite loss", flush=True)
                sys.exit(1)
            if last_micro:
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
        if not rank0:               # rank 0 validates and checkpoints; the others wait
            dist.barrier()
            continue
        val = evaluate(model, vl, device, amp=amp)
        rec = dict(epoch=ep, train_loss=loss_sum / max(1, nb), lr=lr, time=time.time() - t0, **{f"val_{k}": v for k, v in val.items()})
        history.append(rec)
        log(fh, f"ep {ep+1}/{args.epochs} loss={rec['train_loss']:.4f} val_top1={val['top1']:.2f} "
                f"val_f1={val['macro_f1']:.2f} lr={lr:.2e} {rec['time']:.0f}s")
        if val["top1"] > best_val:
            best_val = val["top1"]
            torch.save({"model": model.state_dict(), "epoch": ep, "val": val, "config": cfg}, out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best_val": best_val,
                    "history": history}, last)
        json.dump(history, open(out / "history.json", "w"), indent=1)
        if world > 1:
            dist.barrier()

    if world > 1:                   # final evaluation on rank 0 only
        dist.destroy_process_group()
        if not rank0:
            return
    # ---- final evaluation with the best-val checkpoint ----
    ck = torch.load(out / "best.pt", map_location="cpu")
    model.load_state_dict(ck["model"])
    # per-sample outputs are kept (preds.npz) for paired significance tests, calibration and curves
    test, saved = evaluate(model, tel, device, amp=amp, return_logits=True)
    val_m, val_saved = evaluate(model, vl, device, amp=amp, return_logits=True)
    arrays = {"test_labels": saved["labels"], "test_logits": saved["logits"],
              "val_labels": val_saved["labels"], "val_logits": val_saved["logits"]}
    final = dict(best_epoch=ck["epoch"], best_val_top1=best_val, val=val_m, test=test,
                 pe_params=model.pe_parameters(), params=n_params, config=cfg,
                 train_time_h=sum(h["time"] for h in history) / 3600, history=history)
    log(fh, f"test@{img_size}: {test}")
    if not args.no_multires:
        from viper.data.datasets import build_transforms
        res_results = {}
        # secondary "position interpolation" eval: rotary PEs on absolute patch-unit coordinates
        # (ComRoPE already normalises by the current grid, so it has no separate interpolation mode)
        rotary = args.pe in ("viper", "rope_axial", "rope_mixed", "liere", "string_cayley", "spiral")
        for r in [int(v) for v in args.eval_res.split(",")]:
            _, ev_tf = build_transforms(r, "none", eval_crop_pct=0.875 if dkw["aug"] == "deit" else 1.0)
            te_r = te.__class__(te.base, ev_tf)
            lr_ = DataLoader(te_r, batch_size=128 if r <= 320 else 64, shuffle=False, num_workers=args.workers)
            m_, s_ = evaluate(model, lr_, device, amp=amp, return_logits=True)
            res_results[str(r)] = {"extrapolate": m_}
            arrays[f"res{r}_extrapolate_logits"] = s_["logits"]
            if r != img_size and rotary:
                m_, s_ = evaluate(model, lr_, device, pos_scale=img_size / r, amp=amp, return_logits=True)
                res_results[str(r)]["interpolate"] = m_
                arrays[f"res{r}_interpolate_logits"] = s_["logits"]
            log(fh, f"res {r}: {res_results[str(r)]}")
        final["multires"] = res_results
    np.savez_compressed(out / "preds.npz", **arrays)
    json.dump(final, open(out / "final.json", "w"), indent=2)
    if last.exists():
        last.unlink()               # optimizer state no longer needed; best.pt is kept for analyses
    log(fh, f"FINAL test_top1={test['top1']:.2f} test_f1={test['macro_f1']:.2f}")


if __name__ == "__main__":
    main()
