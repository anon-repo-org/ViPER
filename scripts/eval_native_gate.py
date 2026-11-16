"""Re-evaluate trained ViPER checkpoints with the gate branch at native resolution (gate_res=0).
Writes <run>/multires_native_gate.json; never touches final.json.
--interp: position interpolation instead (positions rescaled to the training range, pos_scale = 224 / r, as in
train.py) -> <run>/multires_native_gate_interp.json."""
import json, sys, torch
from pathlib import Path
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from viper.data.datasets import get_dataset, build_transforms
from viper.models.vit import build_model
from viper.train import evaluate

interp = "--interp" in sys.argv
runs = [a for a in sys.argv[1:] if a != "--interp"]
(tr, va, te, n_cls), dkw = get_dataset("imagenet100", img_size=224, aug="deit")
for run in runs:
    d = ROOT / run
    cfg = json.load(open(d / "config.json"))
    kw = json.loads(cfg["pe_kwargs"]); kw["gate_res"] = 0
    model = build_model(backbone=cfg["backbone"], pe=cfg["pe"], num_classes=n_cls, img_size=224,
                        pretrained=False, drop_path=cfg["drop_path"], pe_kwargs=kw).cuda()
    model.load_state_dict(torch.load(d / "best.pt", map_location="cpu")["model"])
    res = {}
    for r in ((128, 160, 288, 384, 448) if interp else (128, 160, 224, 288, 384, 448)):
        _, tf = build_transforms(r, "none", eval_crop_pct=0.875)
        dl = DataLoader(te.__class__(te.base, tf), batch_size=64, shuffle=False, num_workers=6)
        res[str(r)] = evaluate(model, dl, "cuda", **({"pos_scale": 224 / r} if interp else {}))["top1"]
        print(run, r, round(res[str(r)], 2), flush=True)
    json.dump(res, open(d / ("multires_native_gate_interp.json" if interp else "multires_native_gate.json"), "w"), indent=1)
