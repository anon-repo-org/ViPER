"""
viper.extra_pes
─────────────────────────────────────────────────────────────────
Attention-internal positional encodings: RoPE-Mixed and ALiBi-2D.

These PEs operate INSIDE attention (rotating Q,K for RoPE-Mixed; adding
a bias matrix to attention scores for ALiBi). They cannot be implemented
as input-token modifications like the other PEs in this codebase, so we
monkey-patch the timm DeiT attention modules to apply rotation/bias at
the right point.

Usage:
    from viper.model import DeiTWithCustomPE
    from viper.extra_pes import apply_internal_pe_to_deit

    model = DeiTWithCustomPE(num_classes=8, pe_type='none', ...)
    apply_internal_pe_to_deit(model, pe_type='rope_mixed', H_p=14, W_p=14)
    # OR
    apply_internal_pe_to_deit(model, pe_type='alibi2d', H_p=14, W_p=14)

References:
  - Heo, Park, Han, Yun (ECCV 2024). Rotary Position Embedding for ViT.
  - Press, Smith, Lewis (ICLR 2022). Train Short, Test Long: ALiBi.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


# =============================================================================
# RoPE-Mixed (Heo et al. 2024)
# =============================================================================

class RoPEMixed2D(nn.Module):
    """RoPE-Mixed: learnable per-head 2D rotary position encoding.

    Each attention head has its own learnable frequency vector
    theta in R^{head_dim/4}. The first head_dim/2 channels rotate by
    x-position, the second head_dim/2 channels rotate by y-position.

    Rotation is applied to Q and K only (not V).

    Args:
        num_heads:    number of attention heads
        head_dim:     dimension per head (must be divisible by 4)
        max_h, max_w: informational only (tables computed per forward)
        theta_init:   standard RoPE base frequency
    """
    def __init__(self, num_heads: int, head_dim: int,
                 max_h: int = 64, max_w: int = 64,
                 theta_init: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4"
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.quarter   = head_dim // 4
        self.max_h, self.max_w = max_h, max_w

        freqs_init = 1.0 / (theta_init ** (
            torch.arange(0, self.quarter, dtype=torch.float32) / self.quarter
        ))
        self.freqs = nn.Parameter(
            freqs_init.unsqueeze(0).expand(num_heads, -1).clone()
        )

    def _compute_angles(self, H: int, W: int, device: torch.device):
        """Return (cos_x, sin_x, cos_y, sin_y), each (num_heads, H*W, quarter)."""
        ys = torch.arange(H, device=device, dtype=torch.float32).repeat_interleave(W)
        xs = torch.arange(W, device=device, dtype=torch.float32).repeat(H)
        ang_x = xs.unsqueeze(0).unsqueeze(-1) * self.freqs.unsqueeze(1)
        ang_y = ys.unsqueeze(0).unsqueeze(-1) * self.freqs.unsqueeze(1)
        return (torch.cos(ang_x), torch.sin(ang_x),
                torch.cos(ang_y), torch.sin(ang_y))

    def _rotate(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply 2D rotation. x: (B, H, N, half_dim); cos,sin: (H, N, quarter)."""
        x1, x2 = x[..., :self.quarter], x[..., self.quarter:]
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        return torch.cat([x1 * cos - x2 * sin,
                          x1 * sin + x2 * cos], dim=-1)

    def apply_to_qk(self, q: torch.Tensor, k: torch.Tensor,
                    H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate the patch portion of Q and K. CLS (index 0) unchanged."""
        q_cls, q_pts = q[:, :, :1], q[:, :, 1:]
        k_cls, k_pts = k[:, :, :1], k[:, :, 1:]

        cos_x, sin_x, cos_y, sin_y = self._compute_angles(H, W, q.device)

        half = self.head_dim // 2
        q_pts_x, q_pts_y = q_pts[..., :half], q_pts[..., half:]
        k_pts_x, k_pts_y = k_pts[..., :half], k_pts[..., half:]

        q_pts_x = self._rotate(q_pts_x, cos_x, sin_x)
        q_pts_y = self._rotate(q_pts_y, cos_y, sin_y)
        k_pts_x = self._rotate(k_pts_x, cos_x, sin_x)
        k_pts_y = self._rotate(k_pts_y, cos_y, sin_y)

        q_pts = torch.cat([q_pts_x, q_pts_y], dim=-1)
        k_pts = torch.cat([k_pts_x, k_pts_y], dim=-1)

        q = torch.cat([q_cls, q_pts], dim=2)
        k = torch.cat([k_cls, k_pts], dim=2)
        return q, k


# =============================================================================
# ALiBi-2D
# =============================================================================

class ALiBi2D(nn.Module):
    """2D ALiBi: linear-distance bias for attention scores.

    Bias[i, j] = -m_h * d((x_i, y_i), (x_j, y_j))   for each head h
    Slopes m_h are initialized to ALiBi's recommended values
    (2^{-8*(h+1)/H}) but kept LEARNABLE.
    """
    def __init__(self, num_heads: int, distance: str = "l1"):
        super().__init__()
        assert distance in ("l1", "l2")
        self.num_heads = num_heads
        self.distance  = distance

        slopes_init = torch.tensor(
            [2.0 ** (-8.0 * (h + 1) / num_heads) for h in range(num_heads)],
            dtype=torch.float32,
        )
        self.slopes = nn.Parameter(slopes_init)

    def _compute_distance_matrix(self, H: int, W: int, device: torch.device
                                   ) -> torch.Tensor:
        """(H*W, H*W) pairwise distance matrix."""
        ys = torch.arange(H, device=device).repeat_interleave(W).float()
        xs = torch.arange(W, device=device).repeat(H).float()
        dy = ys.unsqueeze(0) - ys.unsqueeze(1)
        dx = xs.unsqueeze(0) - xs.unsqueeze(1)
        if self.distance == "l1":
            return dy.abs() + dx.abs()
        return torch.sqrt(dy * dy + dx * dx)

    def bias(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """ALiBi bias matrix (num_heads, N+1, N+1) with zero CLS row/col."""
        N = H * W
        d_patches = self._compute_distance_matrix(H, W, device)
        bias_patches = -self.slopes.view(-1, 1, 1) * d_patches.unsqueeze(0)
        bias_full = torch.zeros(self.num_heads, N + 1, N + 1, device=device)
        bias_full[:, 1:, 1:] = bias_patches
        return bias_full


# =============================================================================
# Custom attention wrapper
# =============================================================================

class CustomAttention(nn.Module):
    """Drop-in replacement for timm's Attention that supports internal PE.

    Reuses qkv/proj/dropout from the original attention but inserts a
    RoPE-Mixed rotation (on Q,K) or an ALiBi bias (on attention scores)
    before softmax.
    """
    def __init__(self, original_attn: nn.Module, pe_module: nn.Module,
                 pe_type: str, H_p: int, W_p: int):
        super().__init__()
        self.qkv       = original_attn.qkv
        self.proj      = original_attn.proj
        self.attn_drop = getattr(original_attn, "attn_drop", nn.Identity())
        self.proj_drop = getattr(original_attn, "proj_drop", nn.Identity())
        self.num_heads = original_attn.num_heads
        self.head_dim  = (
            original_attn.head_dim if hasattr(original_attn, "head_dim")
            else (self.qkv.in_features // self.num_heads)
        )
        self.scale = self.head_dim ** -0.5

        self.pe_module = pe_module
        self.pe_type   = pe_type
        self.H_p, self.W_p = H_p, W_p

    def forward(self, x: torch.Tensor, attn_mask=None, is_causal=False
                ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.pe_type == "rope_mixed":
            q, k = self.pe_module.apply_to_qk(q, k, self.H_p, self.W_p)
            attn = (q @ k.transpose(-2, -1)) * self.scale
        elif self.pe_type == "alibi2d":
            attn = (q @ k.transpose(-2, -1)) * self.scale
            # Detect runtime patch-grid size; supports multi-res evaluation.
            N_tokens = attn.shape[-1] - 1            # subtract CLS
            H_rt = int(round(math.sqrt(N_tokens)))
            W_rt = N_tokens // H_rt
            if H_rt * W_rt == N_tokens:
                bias = self.pe_module.bias(H_rt, W_rt, x.device)
                attn = attn + bias.unsqueeze(0)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# =============================================================================
# Patching function
# =============================================================================

def apply_internal_pe_to_deit(model, pe_type: str, H_p: int, W_p: int,
                              num_heads: int = 3, head_dim: int = 64,
                              distance: str = "l1"):
    """Patch a DeiTWithCustomPE model to use RoPE-Mixed or ALiBi-2D.

    Call AFTER constructing the model with pe_type='none'. This builds a
    single shared PE module and swaps each transformer block's attention
    with CustomAttention wrapping the original.

    Args:
        model:     a DeiTWithCustomPE built with pe_type='none'
        pe_type:   'rope_mixed' or 'alibi2d'
        H_p, W_p:  patch grid dimensions
        num_heads: 3 for DeiT-Tiny
        head_dim:  64 for DeiT-Tiny
        distance:  'l1' or 'l2' (ALiBi only)
    """
    if pe_type == "rope_mixed":
        pe_module = RoPEMixed2D(num_heads=num_heads, head_dim=head_dim,
                                 max_h=max(H_p, 64), max_w=max(W_p, 64))
    elif pe_type == "alibi2d":
        pe_module = ALiBi2D(num_heads=num_heads, distance=distance)
    else:
        raise ValueError(f"Unknown internal pe_type: {pe_type}")

    device = next(model.parameters()).device
    pe_module = pe_module.to(device)

    # Track parameters so the optimizer picks them up.
    model.internal_pe = pe_module

    for block in model.backbone.blocks:
        original_attn = block.attn
        block.attn = CustomAttention(
            original_attn=original_attn,
            pe_module=pe_module,
            pe_type=pe_type,
            H_p=H_p, W_p=W_p,
        ).to(device)

    return model
