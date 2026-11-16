#!/usr/bin/env python
"""Params / FLOPs / throughput of every PE on one backbone (single GPU, same batch).

CUDA_VISIBLE_DEVICES=0 python -m scripts.efficiency --backbone deit_small --bs 128 --out results/efficiency_deit_small.json
Throughput = median of 5 passes × 30 timed iterations (after 10 warm-up), CUDA-event timed,
inference in bf16 autocast; also reports the training step time (fwd+bwd+AdamW).
"""
import argparse
import json
import time

import numpy as np
import torch

from viper.models.pe import PE_NAMES
from viper.models.vit import build_model


def timed(fn, iters=30, passes=5):
    for _ in range(10):
        fn()
    res = []
    for _ in range(passes):
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(); res.append((time.perf_counter() - t) / iters)
    return float(np.median(res)), float(np.std(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="deit_small")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--pes", default=",".join(PE_NAMES),
                    help="comma list of <pe> or <name>=<pe>:<json pe_kwargs>, e.g. viper=viper:{\"lepe\":true}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    x = torch.randn(args.bs, 3, args.img, args.img, device="cuda")
    out = {}
    specs = []
    for item in args.pes.split(";" if ";" in args.pes else ","):
        name, _, rest = item.partition("=")
        pe, _, kw = (rest or name).partition(":")
        specs.append((name, pe, json.loads(kw) if kw else {}))
    for name, pe, kw in specs:
        m = build_model(backbone=args.backbone, pe=pe, num_classes=100, img_size=args.img, pe_kwargs=kw).cuda()
        params = sum(p.numel() for p in m.parameters())
        try:
            from fvcore.nn import FlopCountAnalysis
            m.eval()
            fl = FlopCountAnalysis(m, x[:1]); fl.unsupported_ops_warnings(False); fl.uncalled_modules_warnings(False)
            gflops = fl.total() / 1e9
        except Exception as e:  # noqa
            gflops = None
        m.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t_inf, s_inf = timed(lambda: m(x))
        m.train()
        opt = torch.optim.AdamW(m.parameters(), 1e-4)

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(x).float().logsumexp(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        t_tr, s_tr = timed(step, iters=15, passes=3)
        out[name] = dict(params=params, pe_params=m.pe_parameters(), gflops=gflops,
                       infer_img_s=args.bs / t_inf, infer_std=s_inf, train_img_s=args.bs / t_tr,
                       peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)
        del m, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    json.dump(dict(backbone=args.backbone, bs=args.bs, img=args.img, gpu=torch.cuda.get_device_name(), results=out),
              open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
