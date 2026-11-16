#!/usr/bin/env python
"""Does the gain of the gates depend on image content? (results/analysis/texture_gain.json)

python -m scripts.texture_gain                 # ImageNet-100 -> results/analysis/texture_gain.json
python -m scripts.texture_gain imagenet1k      # ImageNet-1k (ViPER vs its control) -> texture_gain_in1k.json

For every ImageNet-100 test image: fine-scale texture energy = mean over pixels of log(1 + |LH|^2+|HL|^2+|HH|^2)
of a one-level db4 DWT of the grey image (test transform, 224^2). Images are split into quintiles of this
energy, and the accuracy of ViPER, its no-gate control and the two gate-source controls is computed per quintile
from the saved test predictions (preds.npz), per seed, at 224^2 and 160^2.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pywt
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from viper.data.datasets import get_dataset  # noqa: E402

RES = os.path.join(ROOT, "results", "in100_deit_small")
METHODS = ["viper", "abl_no_gates", "abl_static_gate", "abl_rgb_gate"]
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def main(dataset="imagenet100"):
    global RES, METHODS
    tag = "" if dataset == "imagenet100" else "_in1k"
    if dataset == "imagenet1k":                                   # same texture measure and quintiles, ImageNet-1k runs
        RES, METHODS = os.path.join(ROOT, "results", "in1k_deit_small"), ["viper", "abl_no_gates"]
    (_, _, te, _), _ = get_dataset(dataset, img_size=224, aug="deit")
    loader = torch.utils.data.DataLoader(te, batch_size=200, shuffle=False, num_workers=8)
    energy, labels = [], []
    for x, y in loader:
        img = x.numpy().transpose(0, 2, 3, 1) * STD + MEAN
        g = img @ np.array([0.299, 0.587, 0.114])
        for gi in g:
            _, (lh, hl, hh) = pywt.dwt2(gi, "db4", mode="reflect")
            energy.append(float(np.log1p((lh ** 2 + hl ** 2 + hh ** 2) * 255 ** 2).mean()))
        labels.append(y.numpy())
    energy, labels = np.array(energy), np.concatenate(labels)
    edges = np.quantile(energy, np.linspace(0, 1, 6))
    q = np.clip(np.searchsorted(edges, energy, side="right") - 1, 0, 4)
    qc = np.zeros_like(q)                                                     # quintiles within each class
    for c_ in np.unique(labels):                                              # (removes the class confound)
        idx = np.where(labels == c_)[0]
        r_ = energy[idx].argsort().argsort()
        qc[idx] = np.minimum(r_ * 5 // len(idx), 4)
    out = {"edges": edges.tolist(), "n_per_bin": np.bincount(q).tolist(), "acc": {}, "acc_within_class": {}}
    for m in METHODS:
        out["acc"][m] = {}
        for s in range(5):
            p = os.path.join(RES, f"{m}_s{s}", "preds.npz")
            if not os.path.exists(p) or not os.path.exists(os.path.join(RES, f"{m}_s{s}", "final.json")):
                continue
            d = np.load(p)
            assert (d["test_labels"] == labels).all(), "test order mismatch"
            out["acc"][m][s], out["acc_within_class"].setdefault(m, {})[s] = {}, {}
            for r, k in (("224", "test_logits"), ("160", "res160_extrapolate_logits")):
                if k not in d.files:
                    continue
                c = d[k].argmax(1) == labels
                out["acc"][m][s][r] = [100 * float(c[q == b].mean()) for b in range(5)]
                out["acc_within_class"][m][s][r] = [100 * float(c[qc == b].mean()) for b in range(5)]
    os.makedirs(os.path.join(ROOT, "results", "analysis"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "analysis", f"texture_gain{tag}.json"), "w"), indent=1)
    for r in ("224", "160"):
        for m in METHODS[0:1] + METHODS[2:]:
            sh = [s for s in out["acc"][m] if s in out["acc"]["abl_no_gates"] and r in out["acc"][m][s]]
            if not sh:
                continue
            d = np.array([[a - b for a, b in zip(out["acc"][m][s][r], out["acc"]["abl_no_gates"][s][r])] for s in sh])
            print(r, m, "gain vs control per quintile (low->high texture):", np.round(d.mean(0), 2), "+-", np.round(d.std(0, ddof=1) / np.sqrt(len(d)), 2))


if __name__ == "__main__":
    main(*sys.argv[1:])
