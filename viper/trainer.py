"""
viper.trainer
─────────────────────────────────────────────────────────────────
Shared training utilities and the two training loops:

  - train_one(...)          : standard fixed-resolution training
  - train_one_multires(...) : per-batch random-resolution training

Both produce identical-shape result dicts so downstream tooling
(eval_multires.py, table extraction) works the same way for either.

Helpers:
  - set_seed             : deterministic seeding for PyTorch/NumPy/Python
  - compute_metrics      : acc, macro-F1, macro-AUC
  - evaluate             : evaluate(model, loader) with loss + metrics
  - build_model          : factory that handles input-only PEs AND
                           attention-internal PEs (rope_mixed, alibi2d)
"""

import importlib.util
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from .viper import ViPERConfig
from .model import DeiTWithCustomPE, make_pe_dynamic
from .extra_pes import apply_internal_pe_to_deit


# DeiT-Tiny constants
DEIT_TINY_NUM_HEADS = 3
DEIT_TINY_HEAD_DIM  = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Which PE types are attention-internal (need extra_pes patching)
INTERNAL_PE_TYPES = ("rope_mixed", "alibi2d")

# Which PE types support dynamic input resolutions (multi-res training)
MULTIRES_SUPPORTED = ("viper", "cpe", "multipeg", "none")


# =============================================================================
# Reproducibility + metrics
# =============================================================================

def set_seed(seed: int):
    """Seed PyTorch, NumPy, and Python's random module."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(logits, labels, num_classes):
    """Return acc, macro-F1, macro-AUC."""
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    preds = probs.argmax(axis=1)
    y = labels.cpu().numpy()
    acc = (preds == y).mean()
    f1  = f1_score(y, preds, average="macro", zero_division=0)
    try:
        auc = (roc_auc_score(y, probs, multi_class="ovr", average="macro")
                if len(np.unique(y)) == num_classes else float("nan"))
    except Exception:
        auc = float("nan")
    return {"acc": float(acc), "f1": float(f1), "auc": float(auc)}


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Compute metrics + average loss over a data loader."""
    model.eval()
    logits_all, labels_all = [], []
    crit = nn.CrossEntropyLoss()
    loss_total = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss_total += crit(out, labels).item()
        logits_all.append(out)
        labels_all.append(labels)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    m = compute_metrics(logits, labels, num_classes)
    m["loss"] = loss_total / len(loader)
    return m


# =============================================================================
# Model factory (handles input-only AND attention-internal PEs)
# =============================================================================

def build_model(pe_type: str, viper_cfg: Optional[ViPERConfig],
                num_classes: int, image_size: int, patch_size: int,
                pretrained: bool = True,
                alibi_distance: str = "l1") -> nn.Module:
    """Build a DeiTWithCustomPE with the requested PE type.

    For attention-internal PEs (rope_mixed, alibi2d) we build the model
    with pe_type='none' and then patch the attention modules.
    """
    if pe_type in INTERNAL_PE_TYPES:
        model = DeiTWithCustomPE(
            num_classes=num_classes, image_size=image_size,
            patch_size=patch_size, pe_type="none", viper_cfg=None,
            pretrained=pretrained,
        )
        H_p = image_size // patch_size
        W_p = image_size // patch_size
        apply_internal_pe_to_deit(
            model, pe_type=pe_type, H_p=H_p, W_p=W_p,
            num_heads=DEIT_TINY_NUM_HEADS, head_dim=DEIT_TINY_HEAD_DIM,
            distance=alibi_distance,
        )
        return model

    return DeiTWithCustomPE(
        num_classes=num_classes, image_size=image_size,
        patch_size=patch_size, pe_type=pe_type, viper_cfg=viper_cfg,
        pretrained=pretrained,
    )


def _count_pe_params(model: nn.Module, pe_type: str) -> int:
    """Count parameters belonging to the candidate PE module."""
    if pe_type in INTERNAL_PE_TYPES:
        return sum(p.numel() for p in model.internal_pe.parameters())
    if pe_type == "viper":
        return sum(p.numel() for p in model.pe_module.parameters())
    if pe_type == "multipeg":
        return sum(p.numel() for p in model.per_block.parameters())
    if model.pe_module is not None:
        return sum(p.numel() for p in model.pe_module.parameters()
                   if p.requires_grad)
    return 0


# =============================================================================
# Data loader plumbing
# =============================================================================

def load_data_loader_module(data_loader_path: str):
    """Import the data_loader.py file from an arbitrary path."""
    spec = importlib.util.spec_from_file_location("dl", data_loader_path)
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    return dl


# =============================================================================
# Standard fixed-resolution training
# =============================================================================

def train_one(pe_type, viper_cfg, train_loader, val_loader, test_loader,
              image_size, patch_size, num_classes,
              n_epochs=30, lr=3e-5, weight_decay=0.05,
              device=DEVICE, seed=42, verbose=True, name=None,
              save_checkpoint_path: Optional[Path] = None,
              pretrained: bool = True,
              alibi_distance: str = "l1"):
    """Train one (pe_type, seed) combination at a fixed resolution."""
    set_seed(seed)

    model = build_model(
        pe_type=pe_type, viper_cfg=viper_cfg,
        num_classes=num_classes, image_size=image_size,
        patch_size=patch_size, pretrained=pretrained,
        alibi_distance=alibi_distance,
    ).to(device)

    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pe_params = _count_pe_params(model, pe_type)

    if verbose:
        tag = name or pe_type
        print(f"[{tag}]  trainable={n_params:,}  PE={pe_params:,}")

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    history = {"train_loss": [], "val": [], "epoch_times": []}
    best_val_acc, best_state = 0.0, None

    for ep in range(n_epochs):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = crit(model(imgs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        sched.step()

        avg_tr = tr_loss / len(train_loader)
        val_m  = evaluate(model, val_loader, device, num_classes)
        ep_t   = time.time() - t0
        history["train_loss"].append(avg_tr)
        history["val"].append(val_m)
        history["epoch_times"].append(ep_t)

        if val_m["acc"] > best_val_acc:
            best_val_acc = val_m["acc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (ep + 1) % max(1, n_epochs // 5) == 0:
            print(f"  ep {ep+1:3d}/{n_epochs}  tr={avg_tr:.4f}  "
                  f"val_acc={val_m['acc']:.4f}  f1={val_m['f1']:.4f}  "
                  f"auc={val_m['auc']:.4f}  t={ep_t:.1f}s")

    test_m = None
    if test_loader is not None and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test_m = evaluate(model, test_loader, device, num_classes)
        if verbose:
            print(f"  TEST  acc={test_m['acc']:.4f}  f1={test_m['f1']:.4f}  "
                  f"auc={test_m['auc']:.4f}")

        if save_checkpoint_path is not None:
            torch.save({
                "state_dict":  {k: v.cpu() for k, v in best_state.items()},
                "pe_type":     pe_type,
                "viper_cfg":   asdict(viper_cfg) if viper_cfg else None,
                "image_size":  image_size,
                "patch_size":  patch_size,
                "num_classes": num_classes,
            }, save_checkpoint_path)

    return {
        "name":         name or pe_type,
        "pe_type":      pe_type,
        "viper_cfg":    asdict(viper_cfg) if viper_cfg else None,
        "seed":         seed,
        "n_params":     n_params,
        "pe_params":    pe_params,
        "best_val_acc": best_val_acc,
        "test":         test_m,
        "history":      history,
        "avg_epoch_t":  float(np.mean(history["epoch_times"])),
    }


# =============================================================================
# Multi-resolution training (per-batch random resolution)
# =============================================================================

def _resize_batch(imgs: torch.Tensor, target_size: int) -> torch.Tensor:
    """Bilinear resize a batch to (target_size, target_size)."""
    return F.interpolate(imgs, size=(target_size, target_size),
                         mode="bilinear", align_corners=False)


def train_one_multires(pe_type, viper_cfg, train_loader, val_loader,
                       test_loader, base_image_size, patch_size, num_classes,
                       train_resolutions: List[int],
                       n_epochs=30, lr=3e-5, weight_decay=0.05,
                       device=DEVICE, seed=42, verbose=True, name=None,
                       save_checkpoint_path: Optional[Path] = None,
                       pretrained: bool = True):
    """Train one (pe_type, seed) with per-batch random-resolution sampling.

    Only PE types in MULTIRES_SUPPORTED are valid (viper, cpe, multipeg, none).
    The model is built at base_image_size, then PE modules are patched to
    handle dynamic resolutions in the forward pass.
    """
    if pe_type not in MULTIRES_SUPPORTED:
        raise ValueError(
            f"pe_type={pe_type} does not support multi-resolution training. "
            f"Use one of: {MULTIRES_SUPPORTED}."
        )

    set_seed(seed)

    model = DeiTWithCustomPE(
        num_classes=num_classes, image_size=base_image_size,
        patch_size=patch_size, pe_type=pe_type, viper_cfg=viper_cfg,
        pretrained=pretrained,
    ).to(device)
    make_pe_dynamic(model)

    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pe_params = _count_pe_params(model, pe_type)

    if verbose:
        tag = name or pe_type
        print(f"[{tag}]  trainable={n_params:,}  PE={pe_params:,}")
        print(f"  multi-res training: {train_resolutions}")

    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    history = {"train_loss": [], "val": [], "epoch_times": [], "res_used": []}
    best_val_acc, best_state = 0.0, None

    rng = random.Random(seed)

    for ep in range(n_epochs):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        ep_resolutions = []
        for imgs, labels in train_loader:
            target_res = rng.choice(train_resolutions)
            ep_resolutions.append(target_res)
            imgs_resized = _resize_batch(imgs.to(device), target_res)
            labels = labels.to(device)

            opt.zero_grad()
            loss = crit(model(imgs_resized), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        sched.step()

        avg_tr = tr_loss / len(train_loader)
        val_m  = evaluate(model, val_loader, device, num_classes)
        ep_t   = time.time() - t0
        history["train_loss"].append(avg_tr)
        history["val"].append(val_m)
        history["epoch_times"].append(ep_t)
        history["res_used"].append(ep_resolutions)

        if val_m["acc"] > best_val_acc:
            best_val_acc = val_m["acc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (ep + 1) % max(1, n_epochs // 5) == 0:
            res_counts = {r: ep_resolutions.count(r) for r in train_resolutions}
            print(f"  ep {ep+1:3d}/{n_epochs}  tr={avg_tr:.4f}  "
                  f"val_acc={val_m['acc']:.4f}  f1={val_m['f1']:.4f}  "
                  f"t={ep_t:.1f}s  res={res_counts}")

    test_m = None
    if test_loader is not None and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test_m = evaluate(model, test_loader, device, num_classes)
        if verbose:
            print(f"  TEST  acc={test_m['acc']:.4f}  f1={test_m['f1']:.4f}  "
                  f"auc={test_m['auc']:.4f}")

        if save_checkpoint_path is not None:
            torch.save({
                "state_dict":         {k: v.cpu() for k, v in best_state.items()},
                "pe_type":            pe_type,
                "viper_cfg":          asdict(viper_cfg) if viper_cfg else None,
                "image_size":         base_image_size,
                "patch_size":         patch_size,
                "num_classes":        num_classes,
                "multires_training":  True,
                "train_resolutions":  train_resolutions,
            }, save_checkpoint_path)

    return {
        "name":              name or pe_type,
        "pe_type":           pe_type,
        "viper_cfg":         asdict(viper_cfg) if viper_cfg else None,
        "seed":              seed,
        "n_params":          n_params,
        "pe_params":         pe_params,
        "train_resolutions": train_resolutions,
        "best_val_acc":      best_val_acc,
        "test":              test_m,
        "history":           history,
        "avg_epoch_t":       float(np.mean(history["epoch_times"])),
    }
