"""
viper.viper
─────────────────────────────────────────────────────────────────
Core ViPER module.

ViPER (Vision Positional Encoding via wavelet Representation) computes
positional codes from a learnable, multi-scale 2D discrete wavelet
transform of the input image. It is a drop-in module that requires no
backbone modification.

This file contains:
  - ViPERConfig: dataclass with the hyperparameters
  - ChannelProjection: RGB-to-grayscale (or learnable) channel reducer
  - ViPERFeatureExtractor: the full PE computation pipeline

Pipeline:
  1. Channel projection (RGB -> 1 channel)
  2. J-level 2D DWT (db4 by default)
  3. Per-subband learnable gating: g_k = sigmoid(W_g e_k) * tanh(W_v e_k)
  4. Gate-broadcast multiplication with each subband's coefficients
  5. Adaptive pool to the patch grid
  6. Sum across subbands and project to d_model via 1x1 conv

Output:  (B, N+1, d_model) — PE tokens ready to add to patch embeddings.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wavelets import DWTForward


# ─── Configuration ───────────────────────────────────────────────────
@dataclass
class ViPERConfig:
    """Default ViPER configuration (matches every paper result)."""
    n_levels:         int  = 3
    wavelet:          str  = "db4"
    channel_mode:     str  = "gray"     # "gray" or "learnable_proj"
    d_pe:             int  = 32          # internal dim of wavelet branch
    use_channel_proj: bool = True
    use_gating:       bool = True


# ─── Channel projection ──────────────────────────────────────────────
class ChannelProjection(nn.Module):
    """Reduce RGB (or any) channels to a single channel for the DWT input."""
    def __init__(self, in_channels: int, mode: str = "gray"):
        super().__init__()
        self.mode = mode
        if mode == "gray":
            self.register_buffer(
                "rgb2gray",
                torch.tensor([0.299, 0.587, 0.114]).view(1, in_channels, 1, 1)
                if in_channels == 3
                else torch.ones(1, in_channels, 1, 1) / in_channels
            )
        elif mode == "learnable_proj":
            self.proj = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
            if in_channels == 3:
                with torch.no_grad():
                    self.proj.weight.copy_(
                        torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
                    )
        else:
            raise ValueError(f"Unknown channel_mode: {mode}")

    def forward(self, x):
        if self.mode == "gray":
            return (x * self.rgb2gray).sum(dim=1, keepdim=True)
        return self.proj(x)


# ─── ViPER feature extractor ─────────────────────────────────────────
class ViPERFeatureExtractor(nn.Module):
    """ViPER's positional code: image -> wavelet features -> patch grid -> d_model.

    Output: (B, N+1, d_model) — to be added to patch+CLS embeddings.
    """
    def __init__(self, cfg: ViPERConfig, in_channels: int,
                 image_h: int, image_w: int, patch_size: int, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.image_h, self.image_w = image_h, image_w
        self.H_p = image_h // patch_size
        self.W_p = image_w // patch_size
        self.d_pe = cfg.d_pe

        # Channel projection
        if cfg.use_channel_proj:
            self.channel_proj = ChannelProjection(in_channels, cfg.channel_mode)
        else:
            self.channel_proj = nn.Identity()

        # 2D-DWT
        self.dwt = DWTForward(J=cfg.n_levels, wave=cfg.wavelet, mode="symmetric")

        # Gating (learnable per-subband)
        self.n_subbands = 1 + 3 * cfg.n_levels       # LL + (LH, HL, HH) per level
        if cfg.use_gating:
            self.scale_embeddings = nn.Parameter(
                torch.randn(self.n_subbands, cfg.d_pe) * 0.02
            )
            self.W_g = nn.Linear(cfg.d_pe, cfg.d_pe, bias=False)
            self.W_v = nn.Linear(cfg.d_pe, cfg.d_pe, bias=False)

        # Final projection from d_pe to d_model
        self.proj_to_dmodel = nn.Linear(cfg.d_pe, d_model)

        # CLS position embedding (learnable, zero-initialized)
        self.cls_pe = nn.Parameter(torch.zeros(1, 1, d_model))

    def _compute_gate(self, sb_idx: int) -> torch.Tensor:
        """Compute the d_pe-dim gating vector for subband sb_idx."""
        if not self.cfg.use_gating:
            device = next(self.parameters()).device
            return torch.ones(self.d_pe, device=device)
        e = self.scale_embeddings[sb_idx]
        g = torch.sigmoid(self.W_g(e))
        v = torch.tanh(self.W_v(e))
        return g * v

    def _modulate(self, coeffs: torch.Tensor, sb_idx: int) -> torch.Tensor:
        """Broadcast-multiply the gate over the subband coefficients."""
        gate = self._compute_gate(sb_idx).view(1, self.d_pe, 1, 1)
        return coeffs * gate

    def _pool_to_patch_grid(self, x):
        """Adaptively pool (or interpolate) to (H_p, W_p)."""
        _, _, h, w = x.shape
        if h == self.H_p and w == self.W_p:
            return x
        if h >= self.H_p and w >= self.W_p:
            return F.adaptive_avg_pool2d(x, (self.H_p, self.W_p))
        return F.interpolate(x, size=(self.H_p, self.W_p),
                              mode="bilinear", align_corners=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B, C, H, W) -> pe_tokens: (B, N+1, d_model)."""
        # Channel projection
        if self.cfg.use_channel_proj:
            x = self.channel_proj(image)
        else:
            x = image.mean(dim=1, keepdim=True)

        # DWT
        Yl, Yh = self.dwt(x)

        # Modulate LL approximation
        ll = self._modulate(Yl, 0)
        ll = self._pool_to_patch_grid(ll)
        accumulator = ll

        # Modulate detail subbands and accumulate
        for j in range(self.cfg.n_levels):
            details = Yh[j]
            for d in range(3):
                sb_idx = 1 + j * 3 + d
                sb = details[:, :, d]
                sb = self._modulate(sb, sb_idx)
                sb = self._pool_to_patch_grid(sb)
                accumulator = accumulator + sb

        # accumulator: (B, d_pe, H_p, W_p) -> (B, N, d_pe) -> (B, N, d_model)
        B = accumulator.shape[0]
        pe_seq = accumulator.permute(0, 2, 3, 1).reshape(B, -1, self.d_pe)
        pe_seq = self.proj_to_dmodel(pe_seq)

        # Prepend CLS PE
        cls = self.cls_pe.expand(B, -1, -1)
        return torch.cat([cls, pe_seq], dim=1)
