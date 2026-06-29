"""
scripts/eval_multires.py — Multi-resolution evaluation of ViPER checkpoints.

Loads each saved checkpoint (trained at a fixed resolution) and evaluates
it at multiple resolutions: {96, 160, 224, 320, 448}.

PE-specific resolution adaptation:
  - 'learned'   : PE table is bilinearly interpolated to the new grid
  - 'relative2d': relative-position embeddings are linearly interpolated
  - 'sincos2d'  : table is rebuilt at the target resolution
  - 'cpe', 'multipeg', 'viper', 'none': handled natively

Usage (from repo root):
    python scripts/eval_multires.py \\
        --dataset bloodmnist \\
        --checkpoint_dir outputs/bloodmnist/checkpoints \\
        --resolutions 96 160 224 320 448

Output:  multires_eval/multires_<dataset>.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Allow `python scripts/eval_multires.py` to import the viper package
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from viper import (
    ViPERConfig, DeiTWithCustomPE,
    evaluate, load_data_loader_module,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Resolution-adaptive PE handling ─────────────────────────────────
def interp_learned_pe(pe_param: torch.Tensor,
                       old_H: int, old_W: int,
                       new_H: int, new_W: int) -> torch.Tensor:
    """Interpolate a (1, old_N+1, d) learned PE table to (1, new_N+1, d).

    The CLS token (index 0) is preserved. Patch positions are reshaped
    to (old_H, old_W) and bilinearly resampled to (new_H, new_W).
    """
    cls_pe   = pe_param[:, :1]
    patch_pe = pe_param[:, 1:]
    d = patch_pe.shape[-1]
    grid = patch_pe.transpose(1, 2).reshape(1, d, old_H, old_W)
    grid = F.interpolate(grid, size=(new_H, new_W),
                          mode="bilinear", align_corners=False)
    new_patch_pe = grid.reshape(1, d, new_H * new_W).transpose(1, 2)
    return torch.cat([cls_pe, new_patch_pe], dim=1)


def interp_rel_embedding(emb_weight: torch.Tensor,
                          old_dim: int, new_dim: int) -> torch.Tensor:
    """Linearly interpolate a (2*old_dim - 1, d) relative-PE embedding."""
    src_size = 2 * old_dim - 1
    dst_size = 2 * new_dim - 1
    src = emb_weight.transpose(0, 1).unsqueeze(0)
    dst = F.interpolate(src, size=dst_size, mode="linear",
                         align_corners=False)
    return dst.squeeze(0).transpose(0, 1)


def load_checkpoint_at_resolution(ckpt_path: Path,
                                    target_resolution: int,
                                    num_classes: int) -> nn.Module:
    """Build a model at target_resolution and load the (adapted) state dict."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pe_type           = ckpt["pe_type"]
    viper_cfg_dict    = ckpt["viper_cfg"]
    patch_size        = ckpt["patch_size"]
    train_image_size  = ckpt["image_size"]
    state_dict        = ckpt["state_dict"]

    viper_cfg = ViPERConfig(**viper_cfg_dict) if viper_cfg_dict else None

    model = DeiTWithCustomPE(
        num_classes=num_classes,
        image_size=target_resolution,
        patch_size=patch_size,
        pe_type=pe_type,
        viper_cfg=viper_cfg,
    )

    old_H, old_W = train_image_size // patch_size, train_image_size // patch_size
    new_H, new_W = target_resolution // patch_size, target_resolution // patch_size

    if pe_type == "learned" and "pe_module.pe" in state_dict:
        state_dict["pe_module.pe"] = interp_learned_pe(
            state_dict["pe_module.pe"], old_H, old_W, new_H, new_W
        )
    elif pe_type == "relative2d":
        if "pe_module.rel_h.weight" in state_dict:
            state_dict["pe_module.rel_h.weight"] = interp_rel_embedding(
                state_dict["pe_module.rel_h.weight"], old_H, new_H
            )
        if "pe_module.rel_w.weight" in state_dict:
            state_dict["pe_module.rel_w.weight"] = interp_rel_embedding(
                state_dict["pe_module.rel_w.weight"], old_W, new_W
            )
    elif pe_type == "sincos2d":
        # Buffer is rebuilt at init for the target resolution; drop stale entry.
        state_dict = {k: v for k, v in state_dict.items()
                       if not k.startswith("pe_module.pe")}

    if "backbone.pos_embed" in state_dict:
        old_pos = state_dict["backbone.pos_embed"]
        if old_pos.shape[1] != new_H * new_W + 1:
            state_dict["backbone.pos_embed"] = torch.zeros(
                1, new_H * new_W + 1, old_pos.shape[-1]
            )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing
                    if k != "backbone.pos_embed"
                    and not (k.startswith("pe_module.pe") and pe_type == "sincos2d")]
    if real_missing:
        print(f"  WARNING: missing keys: {real_missing}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bloodmnist",
                        choices=["eurosat", "bloodmnist", "dermamnist",
                                  "pathmnist", "resisc45", "dtd",
                                  "tissuemnist", "flowers102", "fgvc_aircraft"])
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing .pt checkpoints")
    parser.add_argument("--resolutions", type=int, nargs="+",
                        default=[96, 160, 224, 320, 448])
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_loader_path", default="viper/data_loader.py")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        sys.exit(f"ERROR: checkpoint dir not found: {ckpt_dir}")

    ckpt_files = sorted(ckpt_dir.glob("*.pt"))
    if not ckpt_files:
        sys.exit(f"ERROR: no .pt files in {ckpt_dir}")

    print(f"Found {len(ckpt_files)} checkpoints")
    print(f"Resolutions: {args.resolutions}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_dir = Path(args.out_dir or (ckpt_dir.parent / "multires_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    dl_path = args.data_loader_path
    if not Path(dl_path).exists():
        dl_path = str(REPO_ROOT / args.data_loader_path)
    dl = load_data_loader_module(dl_path)

    first = torch.load(ckpt_files[0], map_location="cpu")
    num_classes = first["num_classes"]
    print(f"Num classes: {num_classes}\n")

    # Cache test loaders by resolution (build once, reuse per checkpoint)
    test_loaders = {}
    for res in args.resolutions:
        print(f"Building test loader at {res}x{res}...")
        _, _, test_loader, nc, _, _ = dl.get_dataset(
            args.dataset, data_root=args.data_root,
            batch_size=args.batch_size, image_size=res,
            seed=42, num_workers=args.num_workers,
        )
        assert nc == num_classes, f"Class count mismatch: {nc} vs {num_classes}"
        test_loaders[res] = test_loader

    print(f"\nEvaluating {len(ckpt_files)} checkpoints "
          f"x {len(args.resolutions)} resolutions...\n")

    all_results = []
    for ckpt_path in ckpt_files:
        ckpt_name = ckpt_path.stem
        print(f"=== {ckpt_name} ===")

        per_res = {}
        for res in args.resolutions:
            try:
                model = load_checkpoint_at_resolution(
                    ckpt_path, res, num_classes
                ).to(DEVICE)
                model.eval()
                metrics = evaluate(model, test_loaders[res], DEVICE, num_classes)
                per_res[res] = metrics
                print(f"  res={res:>3d}  acc={metrics['acc']:.4f}  "
                      f"f1={metrics['f1']:.4f}  auc={metrics['auc']:.4f}")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                print(f"  res={res:>3d}  ERROR: {e}")
                traceback.print_exc()
                per_res[res] = None

        all_results.append({"checkpoint": ckpt_name, "per_resolution": per_res})

    summary_path = out_dir / f"multires_{args.dataset}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {summary_path}")

    print(f"\n{'='*90}")
    print(f"Multi-resolution accuracy table ({args.dataset})")
    print(f"{'='*90}")
    header = f"{'checkpoint':<35}"
    for res in args.resolutions:
        header += f"{res:>10d}"
    print(header)
    print("-" * 90)
    for r in all_results:
        row = f"{r['checkpoint']:<35}"
        for res in args.resolutions:
            m = r["per_resolution"].get(res)
            row += f"{m['acc']:>10.4f}" if m else f"{'-':>10}"
        print(row)


if __name__ == "__main__":
    main()
