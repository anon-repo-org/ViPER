"""Dataset registry. Every builder returns (train_ds, val_ds, test_ds, num_classes).

Large-scale: imagenet100 (CMC 100-class subset, HF parquet `clane9/imagenet-100`), imagenet1k (packed JPEGs),
tiny_imagenet. Domain sets: eurosat, resisc45, bloodmnist, dermamnist,
pathmnist, tissuemnist, dtd, flowers102, fgvc_aircraft (same sources as the
original ViPER code so the numbers stay comparable).

Val/test policy: datasets with an official val+test keep them. ImageNet-100 has
only train/val: we carve a 5%-per-class stratified holdout from train as the
model-selection split and report the official val split as *test*. Datasets
with a single split get a fixed 70/15/15 seed-42 split.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, Subset

ROOT = Path(os.environ.get("VIPER_DATA", Path(__file__).resolve().parents[2] / "data"))
# optional directory with pre-built MedMNIST 224px npz caches (<name>_224.npz); downloaded otherwise
OLD_ROOT = Path(os.environ.get("VIPER_MEDMNIST_NPZ", ROOT / "medmnist_npz"))
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SPLIT_SEED = 42


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def build_transforms(img_size: int, aug: str = "deit", mean=IMAGENET_MEAN, std=IMAGENET_STD,
                     eval_crop_pct: float = 0.875):
    """aug: 'deit' (RandomResizedCrop + flip + RandAugment + RandomErasing),
    'light' (resize + flip + colour jitter; for tiny domain sets), 'none'."""
    from timm.data import create_transform
    if aug == "deit":
        train = create_transform(input_size=img_size, is_training=True, color_jitter=0.3,
                                 auto_augment="rand-m9-mstd0.5-inc1", interpolation="bicubic",
                                 re_prob=0.25, re_mode="pixel", re_count=1, mean=mean, std=std)
    elif aug == "light":
        train = T.Compose([T.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                           T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor(), T.Normalize(mean, std)])
    else:
        train = T.Compose([T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
                           T.ToTensor(), T.Normalize(mean, std)])
    if eval_crop_pct < 1.0:
        evalt = T.Compose([T.Resize(int(img_size / eval_crop_pct), interpolation=T.InterpolationMode.BICUBIC),
                           T.CenterCrop(img_size), T.ToTensor(), T.Normalize(mean, std)])
    else:
        evalt = T.Compose([T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
                           T.ToTensor(), T.Normalize(mean, std)])
    return train, evalt


class TransformDataset(Dataset):
    def __init__(self, base: Dataset, transform: Callable):
        self.base, self.transform = base, transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, y = self.base[i]
        return self.transform(img), int(y)


# --------------------------------------------------------------------------- #
# parquet-backed image datasets (HF snapshots on disk; no `datasets` cache copy)
# --------------------------------------------------------------------------- #
class ParquetImages(Dataset):
    """Reads all parquet shards of a split into memory as JPEG bytes (ImageNet-100
    train is ~8 GB of JPEG bytes: acceptable on this 500 GB machine and avoids
    the random-access cost of Arrow mmap for 100-epoch training)."""

    def __init__(self, files, image_col="image", label_col="label"):
        import pyarrow.parquet as pq
        bytes_, labels = [], []
        for f in sorted(files):
            t = pq.read_table(f, columns=[image_col, label_col])
            col = t.column(image_col).to_pylist()
            bytes_.extend(d["bytes"] if isinstance(d, dict) else d for d in col)
            labels.extend(t.column(label_col).to_pylist())
        self.bytes, self.labels = bytes_, np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return Image.open(io.BytesIO(self.bytes[i])).convert("RGB"), int(self.labels[i])


def stratified_holdout(labels: np.ndarray, frac: float, seed: int = SPLIT_SEED):
    rng = np.random.RandomState(seed)
    hold, keep = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = max(1, int(round(frac * len(idx))))
        hold.extend(idx[:n]); keep.extend(idx[n:])
    return np.array(sorted(keep)), np.array(sorted(hold))


def _parquet_split(root: Path, split: str):
    return sorted(str(p) for p in (root / "data").glob(f"{split}-*.parquet"))


def imagenet100(img_size=224, aug="deit", **_):
    root = ROOT / "hf" / "imagenet-100"
    train = ParquetImages(_parquet_split(root, "train"))
    test = ParquetImages(_parquet_split(root, "validation"))
    tr_tf, ev_tf = build_transforms(img_size, aug)
    keep, hold = stratified_holdout(train.labels, 0.05)
    return (TransformDataset(Subset(train, keep), tr_tf), TransformDataset(Subset(train, hold), ev_tf),
            TransformDataset(test, ev_tf), 100)


class PackedJpegs(Dataset):
    """JPEG bytes packed into one flat file (<split>.bin) with offsets (<split>_off.npy, n+1 entries) and labels
    (<split>_lab.npy); written by scripts/prep_imagenet1k.py. The file is memory-mapped, so all DDP ranks and
    data-loader workers share one copy in the page cache."""

    def __init__(self, prefix: Path):
        self.prefix = Path(prefix)
        self.off = np.load(f"{self.prefix}_off.npy")
        self.labels = np.load(f"{self.prefix}_lab.npy").astype(np.int64)
        self._mm = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self._mm is None:                       # opened lazily in each worker process
            self._mm = np.memmap(f"{self.prefix}.bin", dtype=np.uint8, mode="r")
        b = self._mm[self.off[i]:self.off[i + 1]].tobytes()
        return Image.open(io.BytesIO(b)).convert("RGB"), int(self.labels[i])


def imagenet1k(img_size=224, aug="deit", **_):
    """ImageNet-1k (ILSVRC-2012). Train images stored at 256 px shorter side (scripts/prep_imagenet1k.py),
    val images unchanged. As for ImageNet-100, a stratified 1% holdout of train is the model-selection split
    and the official val split is reported as test."""
    root = ROOT / "imagenet1k"
    train, test = PackedJpegs(root / "train"), PackedJpegs(root / "val")
    tr_tf, ev_tf = build_transforms(img_size, aug)
    keep, hold = stratified_holdout(train.labels, 0.01)
    return (TransformDataset(Subset(train, keep), tr_tf), TransformDataset(Subset(train, hold), ev_tf),
            TransformDataset(test, ev_tf), 1000)


def tiny_imagenet(img_size=64, aug="deit", **_):
    root = ROOT / "hf" / "tiny-imagenet"
    train = ParquetImages(_parquet_split(root, "train"))
    test = ParquetImages(_parquet_split(root, "valid"))
    tr_tf, ev_tf = build_transforms(img_size, aug, eval_crop_pct=1.0)
    keep, hold = stratified_holdout(train.labels, 0.05)
    return (TransformDataset(Subset(train, keep), tr_tf), TransformDataset(Subset(train, hold), ev_tf),
            TransformDataset(test, ev_tf), 200)


# --------------------------------------------------------------------------- #
# domain datasets (same sources as the original ViPER code)
# --------------------------------------------------------------------------- #
def _three_way(ds, seed=SPLIT_SEED, val=0.15, test=0.15):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g).tolist()
    n_te, n_va = int(len(ds) * test), int(len(ds) * val)
    n_tr = len(ds) - n_te - n_va
    return Subset(ds, idx[:n_tr]), Subset(ds, idx[n_tr:n_tr + n_va]), Subset(ds, idx[n_tr + n_va:])


def eurosat(img_size=64, aug="light", **_):
    from torchvision.datasets import EuroSAT
    mean, std = (0.3444, 0.3803, 0.4078), (0.2026, 0.1365, 0.1148)
    tr_tf, ev_tf = build_transforms(img_size, aug, mean, std, eval_crop_pct=1.0)
    base = EuroSAT(root=str(ROOT), download=True)
    tr, va, te = _three_way(base)
    return TransformDataset(tr, tr_tf), TransformDataset(va, ev_tf), TransformDataset(te, ev_tf), 10


class HFDS(Dataset):
    def __init__(self, hf):
        self.hf = hf

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, i):
        it = self.hf[i]
        return it["image"].convert("RGB"), int(it["label"])


def resisc45(img_size=224, aug="light", **_):
    from datasets import load_dataset
    d = load_dataset("timm/resisc45", cache_dir=str(ROOT / "resisc45_hf"))
    tr_tf, ev_tf = build_transforms(img_size, aug, eval_crop_pct=1.0)
    return (TransformDataset(HFDS(d["train"]), tr_tf), TransformDataset(HFDS(d["validation"]), ev_tf),
            TransformDataset(HFDS(d["test"]), ev_tf), 45)


class NPZImages(Dataset):
    def __init__(self, imgs, labels):
        self.imgs, self.labels = imgs, labels.reshape(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return Image.fromarray(self.imgs[i]).convert("RGB"), int(self.labels[i])


def _medmnist(name, n_cls, img_size=224, aug="light", **_):
    import medmnist
    cache = ROOT / "medmnist"
    cache.mkdir(parents=True, exist_ok=True)
    cls = getattr(medmnist, medmnist.INFO[name]["python_class"])
    old = OLD_ROOT / f"{name}_224.npz"
    if old.exists():
        z = np.load(old)
        splits = {s: NPZImages(z[f"{s}_images"], z[f"{s}_labels"]) for s in ("train", "val", "test")}
    else:
        splits = {}
        for s in ("train", "val", "test"):
            ds = cls(split=s, download=True, size=224, root=str(cache))
            splits[s] = NPZImages(ds.imgs, ds.labels)
    tr_tf, ev_tf = build_transforms(img_size, aug, eval_crop_pct=1.0)
    return (TransformDataset(splits["train"], tr_tf), TransformDataset(splits["val"], ev_tf),
            TransformDataset(splits["test"], ev_tf), n_cls)


def bloodmnist(**kw): return _medmnist("bloodmnist", 8, **kw)
def dermamnist(**kw): return _medmnist("dermamnist", 7, **kw)
def pathmnist(**kw): return _medmnist("pathmnist", 9, **kw)
def tissuemnist(**kw): return _medmnist("tissuemnist", 8, **kw)


def dtd(img_size=224, aug="light", **_):
    from torchvision.datasets import DTD
    tr_tf, ev_tf = build_transforms(img_size, aug)
    r = str(ROOT)
    return (TransformDataset(DTD(r, split="train", partition=1, download=True), tr_tf),
            TransformDataset(DTD(r, split="val", partition=1, download=True), ev_tf),
            TransformDataset(DTD(r, split="test", partition=1, download=True), ev_tf), 47)


def flowers102(img_size=224, aug="light", **_):
    from torchvision.datasets import Flowers102
    tr_tf, ev_tf = build_transforms(img_size, aug)
    r = str(ROOT)
    return (TransformDataset(Flowers102(r, split="train", download=True), tr_tf),
            TransformDataset(Flowers102(r, split="val", download=True), ev_tf),
            TransformDataset(Flowers102(r, split="test", download=True), ev_tf), 102)


def fgvc_aircraft(img_size=224, aug="light", **_):
    from torchvision.datasets import FGVCAircraft
    tr_tf, ev_tf = build_transforms(img_size, aug)
    r = str(ROOT)
    return (TransformDataset(FGVCAircraft(r, split="train", download=True), tr_tf),
            TransformDataset(FGVCAircraft(r, split="val", download=True), ev_tf),
            TransformDataset(FGVCAircraft(r, split="test", download=True), ev_tf), 100)


REGISTRY = {
    "imagenet100": imagenet100, "imagenet1k": imagenet1k, "tiny_imagenet": tiny_imagenet, "eurosat": eurosat, "resisc45": resisc45,
    "bloodmnist": bloodmnist, "dermamnist": dermamnist, "pathmnist": pathmnist, "tissuemnist": tissuemnist,
    "dtd": dtd, "flowers102": flowers102, "fgvc_aircraft": fgvc_aircraft,
}
DEFAULTS = {
    "imagenet100": dict(img_size=224, aug="deit"), "imagenet1k": dict(img_size=224, aug="deit"),
    "tiny_imagenet": dict(img_size=64, aug="deit"),
    "eurosat": dict(img_size=224, aug="light"), "resisc45": dict(img_size=224, aug="light"),
    "bloodmnist": dict(img_size=224, aug="light"), "dermamnist": dict(img_size=224, aug="light"),
    "pathmnist": dict(img_size=224, aug="light"), "tissuemnist": dict(img_size=224, aug="light"),
    "dtd": dict(img_size=224, aug="light"), "flowers102": dict(img_size=224, aug="light"),
    "fgvc_aircraft": dict(img_size=224, aug="light"),
}


def get_dataset(name: str, **overrides):
    kw = dict(DEFAULTS[name]); kw.update({k: v for k, v in overrides.items() if v is not None})
    return REGISTRY[name](**kw), kw
