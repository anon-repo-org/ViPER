## ViPER: Vision Positional Encoding with Multiscale Dynamic Wavelet Encoding

**Anonymous WACV 2027 submission - Paper ID 1784**

Code release for the paper *"ViPER: Vision Positional Encoding with Multiscale
Dynamic Wavelet Encoding."*

---

### Repository structure

```
viper-wacv2027/
├── README.md
├── LICENSE                       MIT (post-publication)
├── requirements.txt              Pinned dependencies
├── .gitignore
│
├── viper/                        Library package
│   ├── __init__.py               Public API
│   ├── viper.py                  ViPERConfig + ViPERFeatureExtractor
│   ├── pe_methods.py             Baseline PEs (NoPE, APE, Sin/Cos 2D,
│   │                             Relative-2D, CPE, Multi-PEG)
│   ├── extra_pes.py              Attention-internal PEs (RoPE-Mixed, ALiBi-2D)
│   ├── model.py                  DeiTWithCustomPE wrapper + make_pe_dynamic
│   ├── trainer.py                Training loops, metrics, model factory
│   └── data_loader.py            Unified data loader (9 datasets)
│
└── scripts/
    ├── train.py                  Unified training entry point
    └── eval_multires.py          Resolution generalization evaluation
```

All scripts are run from the **repository root**. They auto-detect the
`viper` package and the data loader at `viper/data_loader.py`.

---

### Environment

Tested with Python 3.10 + CUDA 12.1 on Ubuntu 20.04. Other modern Linux
distributions with a recent NVIDIA GPU should work.

```bash
conda create -n viper python=3.10 -y
conda activate viper
pip install -r requirements.txt
```

### Hardware

All paper results were produced on a single NVIDIA RTX A5000 (24 GB).
Minimum requirement: one GPU with at least **6 GB VRAM**.


### Datasets

The unified loader in `viper/data_loader.py` exposes a single entry point:

```python
from viper.data_loader import get_dataset
train, val, test, num_classes, h, w = get_dataset(
    "bloodmnist", data_root="./data",
    batch_size=64, image_size=224, seed=42,
)
```

| Dataset | Name | Source | Auto-download |
|---|---|---|---|
| EuroSAT | `eurosat` | torchvision | yes |
| BloodMNIST | `bloodmnist` | medmnist | yes |
| DermaMNIST | `dermamnist` | medmnist | yes |
| DTD | `dtd` | torchvision | yes |
| RESISC45 | `resisc45` | HuggingFace `timm/resisc45` | yes |

Each downloads on first use to `data_root` (default `./data`).

---

### Reproducing the paper

All commands are run from the **repository root**.

#### Table 3 - From-scratch comparison

ViT-Tiny trained from scratch on three datasets, 100 epochs each:

```bash
for ds in eurosat bloodmnist dermamnist; do
  python scripts/train.py \
    --dataset $ds --pe all --from_scratch \
    --epochs 100 --seeds 0 1 7 42 123
done
```

#### Table 4 - Extended BloodMNIST comparison (attention-internal PEs)

The five standard PEs from Table 5 plus the two attention-internal ones:

```bash
# Standard input-only PEs (already covered by --pe all in Table 5 run)
python scripts/train.py --dataset bloodmnist --pe all \
    --epochs 50 --seeds 0 1 42 123 7

# Attention-internal PEs (one launch each)
python scripts/train.py --dataset bloodmnist --pe rope_mixed \
    --epochs 50 --seeds 0 1 42 123 7

python scripts/train.py --dataset bloodmnist --pe alibi2d \
    --epochs 50 --seeds 0 1 42 123 7
```

#### Table 5 - Main pretrained comparison (5 datasets × 6 methods × 5 seeds)

```bash
for ds in eurosat bloodmnist dermamnist dtd resisc45; do
  python scripts/train.py \
    --dataset $ds --pe all \
    --epochs 30 --seeds 0 1 7 42 123
done
```

Per-run JSONs land in `outputs/<dataset>/runs/{method}_seed{N}.json`.
Each file contains test accuracy, F1, AUC, per-epoch training history,
parameter counts, and runtime.

#### Table 6 + Figure 5 - Resolution generalization

Two stages: train with `--save_checkpoints`, then evaluate at multiple
resolutions.

```bash
# (1) Train at 224 with checkpoint saving
python scripts/train.py \
    --dataset bloodmnist --pe all \
    --epochs 50 --seeds 0 1 7 42 123 \
    --save_checkpoints

# (2) Evaluate at five resolutions
python scripts/eval_multires.py \
    --dataset bloodmnist \
    --checkpoint_dir outputs/bloodmnist/checkpoints \
    --resolutions 96 160 224 320 448
```

`eval_multires.py` automatically handles PE-specific resolution
adaptation: bilinear interpolation for `learned`, linear interpolation
for `relative2d`, and native handling for `sincos2d`, `cpe`, `multipeg`,
and `viper`.

Output: `outputs/bloodmnist/multires_eval/multires_bloodmnist.json`.

#### Table 7 - Multi-resolution training

A single model trained with resolutions sampled per-batch from the chosen
set, then evaluated at the standard five test resolutions:

```bash
# Train with per-batch random resolution
python scripts/train.py \
    --dataset bloodmnist --pe viper --multires \
    --train_resolutions 160 192 224 256 288 \
    --epochs 50 --seeds 0 1 42 123 7 \
    --save_checkpoints

# (Repeat for cpe, multipeg, none)
for pe in cpe multipeg none; do
  python scripts/train.py \
    --dataset bloodmnist --pe $pe --multires \
    --train_resolutions 160 192 224 256 288 \
    --epochs 50 --seeds 0 1 42 123 7 \
    --save_checkpoints
done

# Then evaluate the multi-res-trained checkpoints
python scripts/eval_multires.py \
    --dataset bloodmnist \
    --checkpoint_dir outputs/bloodmnist_multires/checkpoints
```

### Table 8 - Computational efficiency

The throughput numbers in Table 8 (params, FLOPs, img/s, slowdown) are
measured per PE method on the same DeiT-Tiny backbone at batch size 32.
The measurement loop is documented in the paper; numbers can be
reproduced by timing `model(inputs)` over 200 iterations with 50 warmup
iterations on each `(pe_type, image_size=224, batch=32)` setting.

### Ablations (Section 4.6)

```bash
# Wavelet family
for wav in db1 db2 db4 sym4 coif2; do
  python scripts/train.py \
    --dataset eurosat --pe viper --from_scratch \
    --wavelet $wav --epochs 50 --seeds 0 1 7 42 123
done

# Decomposition depth
for J in 1 2 3 4; do
  python scripts/train.py \
    --dataset eurosat --pe viper --from_scratch \
    --n_levels $J --epochs 50 --seeds 0 1 7 42 123
done

# Internal PE dimension
for d in 16 32 64; do
  python scripts/train.py \
    --dataset eurosat --pe viper --from_scratch \
    --d_pe $d --epochs 50 --seeds 0 1 7 42 123
done
```

After every batch of runs, a summary file `summary_<pe>.json` is also
written at the top level of `outputs/<dataset>/`.



## License

MIT (see `LICENSE`). Released for the purpose of reproducing paper
results during the WACV 2027 review period; a public open-source release
will follow publication.

