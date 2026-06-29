"""
data_loader.py
─────────────────────────────────────────────────────────────────
Unified dataset loaders for ViPER experiments.

Every loader returns a 6-tuple:
    (train_loader, val_loader, test_loader, num_classes, image_h, image_w)

Public entry point:
    get_dataset(name, data_root="./data", **kwargs)

Supported datasets (see DATASET_REGISTRY at the bottom):
    eurosat, resisc45, dtd, flowers102, fgvc_aircraft,
    pathmnist, bloodmnist, dermamnist, tissuemnist
"""

from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import EuroSAT


# ─── Constants ───────────────────────────────────────────────────────
SEED = 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
DEFAULT_VAL_SPLIT  = 0.15
DEFAULT_TEST_SPLIT = 0.15


# ─── Utilities ───────────────────────────────────────────────────────
def make_split(n: int, val_frac: float, test_frac: float, seed: int = SEED):
    """Deterministic shuffle returning (train_idx, val_idx, test_idx)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    n_tr   = n - n_test - n_val
    return idx[:n_tr], idx[n_tr:n_tr + n_val], idx[n_tr + n_val:]


def make_loaders(tr_ds, va_ds, te_ds, batch_size, num_workers=2):
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (DataLoader(tr_ds, shuffle=True,  **kw),
            DataLoader(va_ds, shuffle=False, **kw),
            DataLoader(te_ds, shuffle=False, **kw))


# ─── 1. EuroSAT (satellite, 64×64, 10 classes) ───────────────────────
def get_eurosat(data_root="./data", batch_size=64, image_size=64,
                seed=SEED, num_workers=2):
    mean = (0.3444, 0.3803, 0.4078)   # EuroSAT-specific stats
    std  = (0.2026, 0.1365, 0.1148)

    train_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(0.2, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    full_tr = EuroSAT(root=data_root, transform=train_tf, download=True)
    full_ev = EuroSAT(root=data_root, transform=eval_tf,  download=True)

    tr_i, va_i, te_i = make_split(len(full_tr),
                                  DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT, seed)
    tr_ds = Subset(full_tr, tr_i)
    va_ds = Subset(full_ev, va_i)
    te_ds = Subset(full_ev, te_i)

    return (*make_loaders(tr_ds, va_ds, te_ds, batch_size, num_workers),
            10, image_size, image_size)


# ─── 2. RESISC45 (remote sensing, 256×256, 45 classes) ───────────────
class HFImageDataset(Dataset):
    """Wrap a HuggingFace image dataset to behave like ImageFolder."""
    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB") if hasattr(item["image"], "convert") else item["image"]
        return self.transform(img), int(item["label"])


def get_resisc45(data_root="./data", batch_size=32, image_size=224,
                 seed=SEED, num_workers=2):
    from datasets import load_dataset

    train_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(0.2, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    cache_dir = str(Path(data_root) / "resisc45_hf")
    ds_full = load_dataset("timm/resisc45", cache_dir=cache_dir)

    if "validation" in ds_full and "test" in ds_full:
        tr_ds = HFImageDataset(ds_full["train"],      train_tf)
        va_ds = HFImageDataset(ds_full["validation"], eval_tf)
        te_ds = HFImageDataset(ds_full["test"],       eval_tf)
    else:
        full = ds_full["train"]
        tr_i, va_i, te_i = make_split(len(full),
                                      DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT, seed)
        tr_ds = HFImageDataset(full.select(tr_i), train_tf)
        va_ds = HFImageDataset(full.select(va_i), eval_tf)
        te_ds = HFImageDataset(full.select(te_i), eval_tf)

    return (*make_loaders(tr_ds, va_ds, te_ds, batch_size, num_workers),
            45, image_size, image_size)


# ─── 3. MedMNIST family ──────────────────────────────────────────────
class _MedMNISTWrap(Dataset):
    """MedMNIST returns labels as shape (1,) ndarray; squeeze to int."""
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, int(y[0]) if hasattr(y, "__len__") else int(y)


def get_medmnist(name: str, data_root="./data", batch_size=64,
                 image_size=224, seed=SEED, num_workers=2):
    """
    name: 'pathmnist' | 'bloodmnist' | 'dermamnist' | 'tissuemnist'.
    See https://medmnist.com/ for full list.
    """
    import medmnist
    from medmnist import INFO

    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    n_classes = len(info["label"])

    train_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.1, 0.1, 0.1),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    common = dict(root=data_root, download=True,
                  size=224 if image_size >= 64 else 28)
    tr_full = DataClass(split="train", transform=train_tf, **common)
    va_full = DataClass(split="val",   transform=eval_tf,  **common)
    te_full = DataClass(split="test",  transform=eval_tf,  **common)

    return (*make_loaders(_MedMNISTWrap(tr_full),
                           _MedMNISTWrap(va_full),
                           _MedMNISTWrap(te_full),
                           batch_size, num_workers),
            n_classes, image_size, image_size)


def get_pathmnist(data_root="./data", batch_size=64, image_size=224, **kw):
    return get_medmnist("pathmnist", data_root, batch_size, image_size, **kw)


def get_bloodmnist(data_root="./data", batch_size=64, image_size=224, **kw):
    return get_medmnist("bloodmnist", data_root, batch_size, image_size, **kw)


def get_dermamnist(data_root="./data", batch_size=64, image_size=224, **kw):
    return get_medmnist("dermamnist", data_root, batch_size, image_size, **kw)


def get_tissuemnist(data_root="./data", batch_size=64, image_size=224, **kw):
    """TissueMNIST — kidney microscopy, 8 classes, 236K images."""
    return get_medmnist("tissuemnist", data_root, batch_size, image_size, **kw)


# ─── 4. DTD (Describable Textures, 47 classes) ───────────────────────
def get_dtd(data_root="./data", batch_size=32, image_size=224,
            seed=SEED, num_workers=2):
    """Describable Textures Dataset — 47 classes, ~5,640 images.

    Uses torchvision's built-in DTD with official partition split 1.
    """
    from torchvision.datasets import DTD

    train_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(0.2, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    tr_ds = DTD(root=data_root, split="train", partition=1,
                transform=train_tf, download=True)
    va_ds = DTD(root=data_root, split="val",   partition=1,
                transform=eval_tf,  download=True)
    te_ds = DTD(root=data_root, split="test",  partition=1,
                transform=eval_tf,  download=True)

    return (*make_loaders(tr_ds, va_ds, te_ds, batch_size, num_workers),
            47, image_size, image_size)


# ─── 5. Flowers102 ───────────────────────────────────────────────────
def get_flowers102(data_root="./data", batch_size=32, image_size=224,
                   seed=SEED, num_workers=2):
    """Oxford Flowers-102 — 102 species, ~8K images.

    Note: the standard Flowers102 'train' split has only 1,020 images,
    which is too small for from-scratch ViT training. Following the ViT
    convention, we use 'test' as our training set (6,149 images), and
    'train' (1,020) as our test set.
    """
    from torchvision.datasets import Flowers102

    train_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(0.2, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    tr_ds = Flowers102(root=data_root, split="test",  download=True, transform=train_tf)
    va_ds = Flowers102(root=data_root, split="val",   download=True, transform=eval_tf)
    te_ds = Flowers102(root=data_root, split="train", download=True, transform=eval_tf)

    return (*make_loaders(tr_ds, va_ds, te_ds, batch_size, num_workers),
            102, image_size, image_size)


# ─── 6. FGVC Aircraft ────────────────────────────────────────────────
def get_fgvc_aircraft(data_root="./data", batch_size=32, image_size=224,
                       seed=SEED, num_workers=2):
    """FGVC Aircraft — 100 fine-grained aircraft variants, ~10K images.

    Uses torchvision FGVCAircraft with the 'variant' label (finest level).
    """
    from torchvision.datasets import FGVCAircraft

    train_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    tr_ds = FGVCAircraft(root=data_root, split="train", annotation_level="variant",
                          download=True, transform=train_tf)
    va_ds = FGVCAircraft(root=data_root, split="val",   annotation_level="variant",
                          download=True, transform=eval_tf)
    te_ds = FGVCAircraft(root=data_root, split="test",  annotation_level="variant",
                          download=True, transform=eval_tf)

    return (*make_loaders(tr_ds, va_ds, te_ds, batch_size, num_workers),
            100, image_size, image_size)


# ─── Unified registry ────────────────────────────────────────────────
DATASET_REGISTRY = {
    "eurosat":       get_eurosat,
    "resisc45":      get_resisc45,
    "pathmnist":     get_pathmnist,
    "bloodmnist":    get_bloodmnist,
    "dermamnist":    get_dermamnist,
    "tissuemnist":   get_tissuemnist,
    "dtd":           get_dtd,
    "flowers102":    get_flowers102,
    "fgvc_aircraft": get_fgvc_aircraft,
}

DATASET_DEFAULTS = {
    "eurosat":       dict(image_size=64,  batch_size=64),
    "resisc45":      dict(image_size=224, batch_size=32),
    "pathmnist":     dict(image_size=224, batch_size=64),
    "bloodmnist":    dict(image_size=224, batch_size=64),
    "dermamnist":    dict(image_size=224, batch_size=64),
    "tissuemnist":   dict(image_size=224, batch_size=64),
    "dtd":           dict(image_size=224, batch_size=32),
    "flowers102":    dict(image_size=224, batch_size=32),
    "fgvc_aircraft": dict(image_size=224, batch_size=32),
}


def get_dataset(name: str, data_root="./data", **kwargs):
    """Single entry point: returns (train, val, test, n_cls, h, w).

    Example:
        train, val, test, n_cls, h, w = get_dataset("bloodmnist",
                                                    data_root="./data",
                                                    batch_size=64,
                                                    image_size=224)
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}"
        )
    defaults = dict(DATASET_DEFAULTS[name])
    defaults.update(kwargs)
    defaults["data_root"] = data_root
    return DATASET_REGISTRY[name](**defaults)
