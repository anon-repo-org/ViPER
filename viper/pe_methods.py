"""
viper.pe_methods
─────────────────────────────────────────────────────────────────
Baseline positional encoding methods used as competitors to ViPER.

All "input-only" PEs return a tensor of shape (B, N+1, d_model) to be
ADDED to patch+CLS embeddings before the first transformer block.

CPE and the per-block PEG variant are slightly different: they take the
patch tokens as input and return the modified tokens.

Methods:
  - NoPE          : returns zeros (no positional information)
  - LearnedPE     : learnable absolute table (the original ViT PE)
  - SinCos2DPE    : deterministic 2D sinusoidal extension of Vaswani et al.
  - Relative2DPE  : input-only relative PE derived from row/column embeddings
  - CPE           : single Positional Encoding Generator (CPVT)
  - PerBlockPEG   : per-block PEG (used for Multi-PEG)
"""

import torch
import torch.nn as nn


# ─── NoPE ────────────────────────────────────────────────────────────
class NoPE(nn.Module):
    """Returns zeros — used as the no-positional-information baseline."""
    def __init__(self, num_tokens: int, d_model: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model

    def forward(self, x, image=None):
        return torch.zeros(x.shape[0], self.num_tokens, self.d_model,
                           device=x.device, dtype=x.dtype)


# ─── Learned APE ─────────────────────────────────────────────────────
class LearnedPE(nn.Module):
    """Learnable absolute positional embedding table (the original ViT PE)."""
    def __init__(self, num_tokens: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)

    def forward(self, x, image=None):
        return self.pe.expand(x.shape[0], -1, -1)


# ─── 2D sinusoidal ───────────────────────────────────────────────────
class SinCos2DPE(nn.Module):
    """Deterministic 2D sinusoidal PE.

    The first d_model/2 dims encode the y-axis (one sin/cos pair per
    quarter dim), the second d_model/2 dims encode the x-axis.
    """
    def __init__(self, H: int, W: int, d_model: int):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4"
        d_q = d_model // 4
        y = torch.arange(H, dtype=torch.float32).unsqueeze(1)
        xs = torch.arange(W, dtype=torch.float32).unsqueeze(1)
        wt = 1.0 / (10000 ** (torch.arange(d_q, dtype=torch.float32) / d_q))
        enc_h = torch.cat([torch.sin(y * wt),  torch.cos(y * wt)],  dim=-1)
        enc_w = torch.cat([torch.sin(xs * wt), torch.cos(xs * wt)], dim=-1)
        enc_h = enc_h.unsqueeze(1).expand(H, W, -1)
        enc_w = enc_w.unsqueeze(0).expand(H, W, -1)
        pe = torch.cat([enc_h, enc_w], dim=-1).reshape(1, H * W, d_model)
        cls_pe = torch.zeros(1, 1, d_model)
        self.register_buffer("pe", torch.cat([cls_pe, pe], dim=1))

    def forward(self, x, image=None):
        return self.pe.expand(x.shape[0], -1, -1)


# ─── Input-only relative 2D PE ───────────────────────────────────────
class Relative2DPE(nn.Module):
    """Per-position embedding derived from average relative row/column offsets.

    Unlike attention-internal relative PEs, this one produces a fixed
    additive token like APE — making it directly comparable in our
    drop-in-PE study.
    """
    def __init__(self, H: int, W: int, d_model: int):
        super().__init__()
        self.H, self.W = H, W
        self.rel_h = nn.Embedding(2 * H - 1, d_model // 2)
        self.rel_w = nn.Embedding(2 * W - 1, d_model // 2)
        self.cls_pe = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x, image=None):
        B = x.shape[0]
        device = x.device
        rows = torch.arange(self.H, device=device).repeat_interleave(self.W)
        cols = torch.arange(self.W, device=device).repeat(self.H)
        rel_r = (rows.unsqueeze(1) - rows.unsqueeze(0)) + (self.H - 1)
        rel_c = (cols.unsqueeze(1) - cols.unsqueeze(0)) + (self.W - 1)
        emb_r = self.rel_h(rel_r).mean(dim=1)
        emb_c = self.rel_w(rel_c).mean(dim=1)
        pe = torch.cat([emb_r, emb_c], dim=-1).unsqueeze(0).expand(B, -1, -1)
        cls = self.cls_pe.expand(B, -1, -1)
        return torch.cat([cls, pe], dim=1)


# ─── Conditional PE (single PEG) ─────────────────────────────────────
class CPE(nn.Module):
    """Single-PEG Conditional Positional Encoding (CPVT).

    Applies a depthwise zero-padded conv once to the patch tokens at the
    input. The CPVT paper shows that zero-padding leaks absolute position
    information into the depthwise conv's receptive field.
    """
    def __init__(self, H: int, W: int, d_model: int, k: int = 3):
        super().__init__()
        self.H, self.W = H, W
        self.proj = nn.Conv2d(d_model, d_model, k, padding=k // 2,
                               groups=d_model, padding_mode="zeros")
        self.cls_pe = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x_tokens):
        """x_tokens: (B, N+1, d_model). Returns same shape with PE applied."""
        cls, pts = x_tokens[:, :1], x_tokens[:, 1:]
        B, N, C = pts.shape
        sp = pts.transpose(1, 2).reshape(B, C, self.H, self.W)
        sp = self.proj(sp) + sp                     # CPE: conv output + residual
        pts = sp.flatten(2).transpose(1, 2)
        return torch.cat([cls + self.cls_pe.expand(B, -1, -1), pts], dim=1)


# ─── Per-block PEG (used by Multi-PEG) ───────────────────────────────
class PerBlockPEG(nn.Module):
    """A single PEG conv applied at one transformer block.

    DeiTWithCustomPE holds a ModuleList of these for the multipeg variant.
    """
    def __init__(self, H: int, W: int, d_model: int, k: int = 3):
        super().__init__()
        self.H, self.W = H, W
        self.conv = nn.Conv2d(d_model, d_model, k, padding=k // 2,
                               groups=d_model, padding_mode="zeros")

    def forward(self, x_tokens):
        cls, pts = x_tokens[:, :1], x_tokens[:, 1:]
        B, N, C = pts.shape
        sp = pts.transpose(1, 2).reshape(B, C, self.H, self.W)
        sp = self.conv(sp) + sp
        pts = sp.flatten(2).transpose(1, 2)
        return torch.cat([cls, pts], dim=1)
