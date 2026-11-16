"""Re-evaluate finished ADE20K runs (final.pt) for extra metrics: mAcc (mean per-class pixel accuracy),
mDice (mean per-class F1), fwIoU, plus per-class acc. Same sliding-window protocol as viper.seg.evaluate.

  python -m scripts.seg_reeval results/ade20k_deit_small/*_s?
Writes <run>/extra_metrics.json (skips runs that already have it).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent         # scripts/queue.py would shadow the stdlib queue module
sys.path = [q for q in sys.path if Path(q or ".").resolve() != _HERE]
sys.path.insert(0, str(_HERE.parent))

import json  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from viper.models.vit import build_model  # noqa: E402
from viper.seg import ADE20K, IGNORE, N_CLS, SegViT, slide_logits  # noqa: E402


@torch.no_grad()
def metrics(model, ds, device):
    model.eval()
    inter = torch.zeros(N_CLS, dtype=torch.float64)
    area_p = torch.zeros(N_CLS, dtype=torch.float64)
    area_t = torch.zeros(N_CLS, dtype=torch.float64)
    for i in range(len(ds)):
        x, y = ds[i]
        lg = slide_logits(model, x[None].to(device))
        lg = F.interpolate(lg, size=y.shape, mode="bilinear", align_corners=False)
        pred = lg.argmax(1)[0].cpu()
        m = y != IGNORE
        p, t = pred[m], y[m]
        inter += torch.bincount(t[p == t], minlength=N_CLS).double()
        area_p += torch.bincount(p, minlength=N_CLS).double()
        area_t += torch.bincount(t, minlength=N_CLS).double()
    union = area_p + area_t - inter
    iou = inter / union.clamp(min=1)
    acc = inter / area_t.clamp(min=1)
    dice = 2 * inter / (area_p + area_t).clamp(min=1)
    v, vt = union > 0, area_t > 0
    return dict(miou=100 * iou[v].mean().item(), aacc=100 * (inter.sum() / area_t.sum()).item(),
                macc=100 * acc[vt].mean().item(), mdice=100 * dice[v].mean().item(),
                fwiou=100 * ((area_t / area_t.sum()) * iou).sum().item(),
                acc=[round(100 * a, 2) for a in acc.tolist()])


def main():
    device = "cuda"
    va = ADE20K("val", 512, train=False)
    for run in map(Path, sys.argv[1:]):
        outf = run / "extra_metrics.json"
        if outf.exists() or not (run / "final.pt").exists():
            continue
        icfg = json.load(open(run / "config.json"))["init_config"]
        vit = build_model(backbone=icfg["backbone"], pe=icfg["pe"], num_classes=icfg["n_cls"],
                          img_size=icfg["img_size"], pretrained=False, drop_path=icfg["drop_path"],
                          pe_kwargs=json.loads(icfg["pe_kwargs"]))
        model = SegViT(vit).to(device)
        model.load_state_dict(torch.load(run / "final.pt", map_location="cpu")["model"])
        r = metrics(model, va, device)
        old = json.load(open(run / "final.json"))["val"]["miou"]
        assert abs(r["miou"] - old) < 0.05, (run, r["miou"], old)   # same protocol reproduces the stored mIoU
        json.dump(r, open(outf, "w"))
        print(f"{run.name}: mIoU {r['miou']:.2f} mAcc {r['macc']:.2f} mDice {r['mdice']:.2f} fwIoU {r['fwiou']:.2f}",
              flush=True)


if __name__ == "__main__":
    main()
