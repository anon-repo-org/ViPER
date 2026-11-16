#!/usr/bin/env python
"""Out-of-distribution evaluation of the ImageNet-100 models (no training).

python -m scripts.robust_eval cache          # decode the IN-100 classes of each test set once -> data/robust/*.pt
python -m scripts.robust_eval eval [runs...] # evaluate checkpoints     -> results/robust/<method>_s<seed>.json

Test sets (clip-benchmark webdataset copies on Hugging Face): ImageNet-v2 matched-frequency, ImageNet-Sketch,
ImageNet-R and ImageNet-A. Only images whose class is one of the 100 ImageNet-100 classes are kept, and the
models predict among those 100 classes (as on the in-distribution test set). ImageNet-v2/Sketch cover all 1000
ImageNet classes, ImageNet-R/A only 200, so the latter use the overlapping classes (counts in the json).
Class matching: ImageNet-100 names are ImageNet synset descriptions -> ImageNet-1k index via timm's ImageNetInfo;
webdataset labels are ImageNet-1k indices (v2, Sketch) or indices into the set's own 200 CLIP-style class names,
mapped to ImageNet-1k through the 1000 CLIP-style names of the v2 copy. Preprocessing = the test transform.
"""
from __future__ import annotations

import glob
import io
import json
import os
import re
import sys
import tarfile

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "robust")
SETS = {"v2": "wds_imagenetv2", "sketch": "wds_imagenet_sketch", "r": "wds_imagenet-r", "a": "wds_imagenet-a"}


def in100_to_1k():
    from timm.data import ImageNetInfo
    info = ImageNetInfo()
    desc = {info.index_to_description(i): i for i in range(1000)}
    import yaml
    front = open(os.path.join(ROOT, "data/hf/imagenet-100/README.md")).read().split("---")[1]
    names = yaml.safe_load(front)["dataset_info"]["features"][1]["dtype"]["class_label"]["names"]
    return {desc[n.strip()]: int(j) for j, n in names.items()}         # 1k index -> IN-100 label


def cache():
    from viper.data.datasets import build_transforms
    _, tf = build_transforms(224, "none", eval_crop_pct=0.875)
    m1k = in100_to_1k()
    assert len(m1k) == 100, len(m1k)
    clip1k = [l.strip() for l in open(os.path.join(DATA, SETS["v2"], "classnames.txt"))]
    for key, d in SETS.items():
        names = [l.strip() for l in open(os.path.join(DATA, d, "classnames.txt"))]
        to1k = (lambda c: c) if len(names) == 1000 else (lambda c, n=names: clip1k.index(n[c]))
        xs, ys = [], []
        for tf_path in sorted(glob.glob(os.path.join(DATA, d, "test", "*.tar")), key=lambda p: int(os.path.basename(p)[:-4])):
            tar = tarfile.open(tf_path)
            cur = {}
            for mem in tar.getmembers():
                stem, ext = os.path.splitext(mem.name)
                cur.setdefault(stem, {})[ext] = mem
            for stem, rec in cur.items():
                if ".cls" not in rec:
                    continue
                c = int(tar.extractfile(rec[".cls"]).read().decode().strip())
                k = to1k(c)
                if k not in m1k:
                    continue
                img_m = next(v for e, v in rec.items() if e != ".cls")
                img = Image.open(io.BytesIO(tar.extractfile(img_m).read())).convert("RGB")
                xs.append(tf(img).half()); ys.append(m1k[k])
        x, y = torch.stack(xs), torch.tensor(ys)
        torch.save(dict(x=x, y=y), os.path.join(DATA, f"{key}_in100.pt"))
        print(key, "images", len(y), "classes", len(set(ys)), flush=True)


@torch.no_grad()
def evaluate(runs):
    from viper.models.vit import build_model
    data = {k: torch.load(os.path.join(DATA, f"{k}_in100.pt")) for k in SETS if os.path.exists(os.path.join(DATA, f"{k}_in100.pt"))}
    os.makedirs(os.path.join(ROOT, "results", "robust"), exist_ok=True)
    for run in runs:
        name = os.path.basename(run.rstrip("/"))
        dst = os.path.join(ROOT, "results", "robust", name + ".json")
        if os.path.exists(dst):
            continue
        cfg = json.load(open(os.path.join(run, "config.json")))
        m = build_model(backbone=cfg["backbone"], pe=cfg["pe"], num_classes=cfg["n_cls"], img_size=cfg["img_size"],
                        pretrained=False, drop_path=0.0, pe_kwargs=json.loads(cfg["pe_kwargs"]))
        m.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location="cpu")["model"])
        m = m.cuda().eval()
        res = {}
        for k, d in data.items():
            correct = 0
            for i in range(0, len(d["y"]), 128):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = m(d["x"][i:i + 128].cuda().float()).argmax(1).cpu()
                correct += (p == d["y"][i:i + 128]).sum().item()
            res[k] = dict(top1=100 * correct / len(d["y"]), n=len(d["y"]), n_cls=int(d["y"].unique().numel()))
        json.dump(res, open(dst, "w"), indent=1)
        print(name, {k: round(v["top1"], 2) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "cache":
        cache()
    else:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from make_tables import PE_INFO                                   # the methods in the paper only
        keep = set(PE_INFO) | {"abl_no_gates"}
        runs = sys.argv[2:] or sorted(d for d in glob.glob(os.path.join(ROOT, "results", "in100_deit_small", "*_s[0-4]"))
                                      if os.path.exists(os.path.join(d, "final.json"))      # finished runs only
                                      and os.path.basename(d).rsplit("_s", 1)[0] in keep)
        evaluate(runs)
