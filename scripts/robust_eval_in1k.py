#!/usr/bin/env python
"""Out-of-distribution evaluation of the ImageNet-1k models (no training).

python -m scripts.robust_eval_in1k [--device cuda] [--limit N] [runs...]   -> results/robust_in1k/<run>.json

Same four test sets as scripts/robust_eval.py (clip-benchmark webdataset copies in data/robust/wds_*), but with
all 1000 classes: ImageNet-v2 matched-frequency and ImageNet-Sketch are scored 1000-way; ImageNet-R and
ImageNet-A cover 200 classes and are scored over those 200 classes (logits of the other 800 masked), the
standard protocol of their papers. The 200 CLIP-style class names are mapped to ImageNet-1k indices through
the 1000 names of the v2 copy, as in robust_eval.py. Preprocessing = the ImageNet-1k test transform.
Every image is decoded once and fed to all models, so the whole sweep is a single pass over the data.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import tarfile

import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "robust")
SETS = {"v2": "wds_imagenetv2", "sketch": "wds_imagenet_sketch", "r": "wds_imagenet-r", "a": "wds_imagenet-a"}
OUT = os.path.join(ROOT, "results", "robust_in1k")


def class_map(key):
    """label of the set -> ImageNet-1k index"""
    names = [l.strip() for l in open(os.path.join(DATA, SETS[key], "classnames.txt"))]
    if len(names) == 1000:
        return list(range(1000))
    clip1k = [l.strip() for l in open(os.path.join(DATA, SETS["v2"], "classnames.txt"))]
    return [clip1k.index(n) for n in names]


class TarImages(torch.utils.data.IterableDataset):
    def __init__(self, key, tf, limit=0):
        self.tars = sorted(glob.glob(os.path.join(DATA, SETS[key], "test", "*.tar")),
                           key=lambda p: int(os.path.basename(p)[:-4]))
        self.to1k, self.tf, self.limit = class_map(key), tf, limit

    def __iter__(self):
        w = torch.utils.data.get_worker_info()
        tars = self.tars if w is None else self.tars[w.id::w.num_workers]
        n = 0
        for p in tars:
            tar = tarfile.open(p)
            cur = {}
            for mem in tar.getmembers():
                stem, ext = os.path.splitext(mem.name)
                cur.setdefault(stem, {})[ext] = mem
            for stem, rec in cur.items():
                if ".cls" not in rec:
                    continue
                c = int(tar.extractfile(rec[".cls"]).read().decode().strip())
                img_m = next(v for e, v in rec.items() if e != ".cls")
                img = Image.open(io.BytesIO(tar.extractfile(img_m).read())).convert("RGB")
                yield self.tf(img), self.to1k[c]
                n += 1
                if self.limit and n >= self.limit:
                    return


def load(run, device):
    from viper.models.vit import build_model
    cfg = json.load(open(os.path.join(run, "config.json")))
    m = build_model(backbone=cfg["backbone"], pe=cfg["pe"], num_classes=cfg["n_cls"], img_size=cfg["img_size"],
                    pretrained=False, drop_path=0.0, pe_kwargs=json.loads(cfg["pe_kwargs"]))
    m.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location="cpu")["model"])
    return m.to(device).eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="images per set and worker (smoke test)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()
    from viper.data.datasets import build_transforms
    _, tf = build_transforms(224, "none")                       # = the ImageNet-1k test transform
    runs = args.runs
    if not runs:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from make_tables import PE_INFO                          # the methods in the paper only
        keep = set(PE_INFO) | {"abl_no_gates", "viper_glr"}
        runs = sorted(d for d in glob.glob(os.path.join(ROOT, "results", "in1k_deit_small", "*_s[0-4]"))
                      if os.path.exists(os.path.join(d, "final.json"))
                      and os.path.basename(d).rsplit("_s", 1)[0] in keep)
    os.makedirs(OUT, exist_ok=True)
    runs = [r for r in runs if args.limit or not os.path.exists(os.path.join(OUT, os.path.basename(r.rstrip("/")) + ".json"))]
    if not runs:
        print("nothing to do"); return
    models = {os.path.basename(r.rstrip("/")): load(r, args.device) for r in runs}
    print("models:", list(models), flush=True)
    res = {n: {} for n in models}
    for key in SETS:
        cls = sorted(set(class_map(key)))
        mask = torch.full((1000,), float("-inf"), device=args.device)
        mask[cls] = 0
        dl = torch.utils.data.DataLoader(TarImages(key, tf, args.limit), batch_size=args.bs,
                                         num_workers=args.workers, pin_memory=args.device == "cuda")
        correct, total = {n: 0 for n in models}, 0
        for x, y in dl:
            x, y = x.to(args.device, non_blocking=True), y.to(args.device)
            total += len(y)
            for n, m in models.items():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                    logit = m(x).float()
                correct[n] += ((logit + mask).argmax(1) == y).sum().item()
        for n in models:
            res[n][key] = dict(top1=100 * correct[n] / total, n=total, n_cls=len(cls))
        print(key, total, {n: round(100 * c / total, 2) for n, c in correct.items()}, flush=True)
    if args.limit:
        return
    for n, r in res.items():
        r["avg"] = sum(v["top1"] for v in r.values()) / len(SETS)
        json.dump(r, open(os.path.join(OUT, n + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
