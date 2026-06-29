"""
scripts/train.py — Unified training entry point for ViPER experiments.

This single script handles every training run in the paper:
  - Standard fixed-resolution training (Table 3, Table 5)
  - Attention-internal PEs RoPE-Mixed and ALiBi-2D (Table 4)
  - Multi-resolution training (Table 7)

The PE choice is set by --pe; the training mode is set by --multires
(off by default). Pass --pe all to run every PE in the standard
five-method roster on a single dataset.

Usage examples
--------------
# Single PE, single dataset, three seeds (pretrained DeiT-Tiny, 30 epochs)
python scripts/train.py --dataset bloodmnist --pe viper \\
    --epochs 30 --seeds 42 123 7

# Run the full five-method roster (Table 5)
python scripts/train.py --dataset bloodmnist --pe all \\
    --epochs 30 --seeds 0 1 7 42 123

# From-scratch training (Table 3)
python scripts/train.py --dataset eurosat --pe all --from_scratch \\
    --epochs 100 --seeds 0 1 7 42 123

# Attention-internal PEs (Table 4)
python scripts/train.py --dataset bloodmnist --pe rope_mixed \\
    --epochs 30 --seeds 42 123 7

# Multi-resolution training (Table 7)
python scripts/train.py --dataset bloodmnist --pe viper --multires \\
    --train_resolutions 160 192 224 256 288 \\
    --epochs 30 --seeds 42 123 7

# Save checkpoints for later multi-res evaluation (Table 6, Figure 5)
python scripts/train.py --dataset bloodmnist --pe all \\
    --epochs 30 --seeds 0 1 7 42 123 --save_checkpoints
"""

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Allow `python scripts/train.py` to import the viper package
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from viper import (
    ViPERConfig,
    train_one, train_one_multires,
    load_data_loader_module,
    INTERNAL_PE_TYPES, MULTIRES_SUPPORTED,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# All PE types the script can train. "all" expands to the five-method roster
# used in Table 5 of the paper (matches the original `--suite main`).
PE_TYPES_STANDARD = [
    "none", "learned", "sincos2d", "relative2d", "cpe", "multipeg", "viper",
]
PE_TYPES_INTERNAL = list(INTERNAL_PE_TYPES)
PE_TYPES_ALL = PE_TYPES_STANDARD + PE_TYPES_INTERNAL

# Five-method roster for the main paper comparison (Table 5)
TABLE5_ROSTER = ["none", "learned", "sincos2d", "cpe", "multipeg", "viper"]


def default_viper_cfg() -> ViPERConfig:
    """The configuration used for every paper result (8,896 PE parameters)."""
    return ViPERConfig(n_levels=3, wavelet="db4", channel_mode="gray", d_pe=32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified training entry point for ViPER experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset / loader
    parser.add_argument("--dataset", default="bloodmnist",
                        choices=["eurosat", "bloodmnist", "dermamnist",
                                 "pathmnist", "resisc45", "dtd",
                                 "tissuemnist", "flowers102", "fgvc_aircraft"])
    parser.add_argument("--data_loader_path", default="viper/data_loader.py",
                        help="Path to data_loader.py (relative to repo root)")
    parser.add_argument("--data_root", default="./data",
                        help="Where to cache downloaded datasets")
    parser.add_argument("--num_workers", type=int, default=2)

    # Model
    parser.add_argument("--pe", default="viper",
                        choices=PE_TYPES_ALL + ["all"],
                        help="Which PE to train. 'all' runs the five-method "
                             "roster (none, learned, sincos2d, cpe, multipeg, viper).")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image size (224 matches DeiT-Tiny pretraining)")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Patch size (16 for DeiT-Tiny)")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Train ViT-Tiny from scratch instead of "
                             "loading pretrained DeiT-Tiny weights")

    # ViPER overrides (also used as ablation knobs)
    parser.add_argument("--wavelet", default=None,
                        help="Override ViPER wavelet family (e.g., db1, sym4)")
    parser.add_argument("--n_levels", type=int, default=None,
                        help="Override ViPER decomposition depth J")
    parser.add_argument("--d_pe", type=int, default=None,
                        help="Override ViPER internal PE dimension")

    # Attention-internal PE specific
    parser.add_argument("--alibi_distance", default="l1", choices=["l1", "l2"],
                        help="Distance metric for ALiBi-2D")

    # Optimization
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    # Multi-resolution training (Table 7)
    parser.add_argument("--multires", action="store_true",
                        help="Train with per-batch random resolutions. "
                             "Only valid for pe types in MULTIRES_SUPPORTED.")
    parser.add_argument("--train_resolutions", type=int, nargs="+",
                        default=[160, 192, 224, 256, 288],
                        help="Resolutions to sample from during multi-res training")

    # Output / checkpoints
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (defaults to outputs/<dataset>)")
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="Save best checkpoints for later multi-res eval")

    return parser.parse_args()


def resolve_pe_list(pe_arg: str) -> list:
    """Convert the --pe argument into a list of pe_types to train."""
    if pe_arg == "all":
        return TABLE5_ROSTER
    return [pe_arg]


def make_viper_cfg(args) -> ViPERConfig:
    """Build a ViPERConfig, applying any CLI overrides for ablations."""
    cfg = default_viper_cfg()
    if args.wavelet is not None:
        cfg.wavelet = args.wavelet
    if args.n_levels is not None:
        cfg.n_levels = args.n_levels
    if args.d_pe is not None:
        cfg.d_pe = args.d_pe
    return cfg


def main():
    args = parse_args()

    # Output directory
    suffix = "_multires" if args.multires else ("_fs" if args.from_scratch else "")
    out_dir = Path(args.out_dir or f"outputs/{args.dataset}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    if args.save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}  image_size={args.image_size}  "
          f"patch_size={args.patch_size}")
    if args.from_scratch:
        print("Mode: from-scratch (random ViT-Tiny init)")
    if args.multires:
        print(f"Multi-resolution training: {args.train_resolutions}")
    print(f"Outputs: {out_dir.resolve()}")

    # Validate the PE list against the training mode
    pe_list = resolve_pe_list(args.pe)
    if args.multires:
        bad = [pe for pe in pe_list if pe not in MULTIRES_SUPPORTED]
        if bad:
            sys.exit(f"ERROR: --multires requires PE in {MULTIRES_SUPPORTED}, got {bad}")

    # Load data
    dl_path = args.data_loader_path
    if not Path(dl_path).exists():
        dl_path = str(REPO_ROOT / args.data_loader_path)
    dl = load_data_loader_module(dl_path)
    os.makedirs(args.data_root, exist_ok=True)
    train_loader, val_loader, test_loader, num_classes, img_h, img_w = \
        dl.get_dataset(args.dataset,
                       data_root=args.data_root,
                       batch_size=args.batch_size,
                       image_size=args.image_size,
                       seed=args.seeds[0],
                       num_workers=args.num_workers)
    print(f"  {num_classes} classes, {img_h}x{img_w}")
    print(f"  train={len(train_loader.dataset):,}  "
          f"val={len(val_loader.dataset):,}  "
          f"test={len(test_loader.dataset):,}")

    # Build the run list (one entry per (pe_type, seed))
    viper_cfg_default = make_viper_cfg(args)
    runs = []
    for pe in pe_list:
        cfg = viper_cfg_default if pe == "viper" else None
        # Pick a name reflecting any ViPER overrides (used in JSON filenames)
        if pe == "viper" and (args.wavelet or args.n_levels or args.d_pe):
            suffix_parts = []
            if args.wavelet:  suffix_parts.append(f"wav{args.wavelet}")
            if args.n_levels: suffix_parts.append(f"J{args.n_levels}")
            if args.d_pe:     suffix_parts.append(f"d{args.d_pe}")
            run_basename = "viper_" + "_".join(suffix_parts)
        else:
            run_basename = pe
        runs.append((pe, cfg, run_basename))

    # Train
    all_results = []
    for seed in args.seeds:
        for pe_type, cfg, basename in runs:
            run_id = f"{basename}_seed{seed}"
            run_path = runs_dir / f"{run_id}.json"
            if run_path.exists():
                print(f"SKIP {run_id} (already done)")
                with open(run_path) as f:
                    all_results.append(json.load(f))
                continue

            print(f"\n{'='*70}\n{run_id}\n{'='*70}")
            if cfg:
                print(f"  cfg: {cfg}")

            ckpt_path = (ckpt_dir / f"{run_id}.pt") if args.save_checkpoints else None

            try:
                if args.multires:
                    r = train_one_multires(
                        pe_type, cfg,
                        train_loader, val_loader, test_loader,
                        base_image_size=args.image_size,
                        patch_size=args.patch_size,
                        num_classes=num_classes,
                        train_resolutions=args.train_resolutions,
                        n_epochs=args.epochs,
                        lr=args.lr, weight_decay=args.weight_decay,
                        device=DEVICE, seed=seed, name=run_id,
                        save_checkpoint_path=ckpt_path,
                        pretrained=not args.from_scratch,
                    )
                else:
                    r = train_one(
                        pe_type, cfg,
                        train_loader, val_loader, test_loader,
                        image_size=args.image_size,
                        patch_size=args.patch_size,
                        num_classes=num_classes,
                        n_epochs=args.epochs,
                        lr=args.lr, weight_decay=args.weight_decay,
                        device=DEVICE, seed=seed, name=run_id,
                        save_checkpoint_path=ckpt_path,
                        pretrained=not args.from_scratch,
                        alibi_distance=args.alibi_distance,
                    )
                all_results.append(r)
                with open(run_path, "w") as f:
                    json.dump(r, f, indent=2, default=str)
            except Exception as e:
                import traceback
                print(f"[ERROR] {run_id}: {e}")
                traceback.print_exc()

    # Summary table
    summary_path = out_dir / f"summary_{args.pe}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*70}\nSummary saved: {summary_path}\n{'='*70}")
    print(f"{'name':<35} {'pe_type':<12} {'test_acc':>10} {'test_f1':>10} {'PE params':>10}")
    print("-" * 82)
    rows = [(r["name"], r["pe_type"],
             (r["test"]["acc"] if r["test"] else float("nan")),
             (r["test"]["f1"] if r["test"] else float("nan")),
             r["pe_params"])
            for r in all_results]
    rows.sort(key=lambda x: -x[2] if not math.isnan(x[2]) else 0)
    for name, pe_type, acc, f1, pe_p in rows:
        print(f"{name:<35} {pe_type:<12} {acc:>10.4f} {f1:>10.4f} {pe_p:>10,}")


if __name__ == "__main__":
    main()
