"""Attention block that consumes a shared `PEContext` (rotary tables, per-token
gates, additive biases, LePE). Parameter names match timm's `Attention` so the
pretrained weights load with `load_state_dict`."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pe import PEContext, apply_rotary


class PEAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, qk_norm: bool = False,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, norm_layer=nn.LayerNorm,
                 ctx: Optional[PEContext] = None, layer_idx: int = 0, lepe_kernel: int = 3):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ctx = ctx
        self.layer_idx = layer_idx
        # LePE conv (only used when ctx.lepe); depthwise on the V grid
        self.lepe_conv = nn.Conv2d(dim, dim, lepe_kernel, 1, lepe_kernel // 2, groups=dim)
        self.last_attn = None          # filled when ctx.extra['keep_attn'] is set (analysis)

    def forward(self, x: torch.Tensor, attn_mask=None, is_causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        ctx = self.ctx
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if ctx is not None and ctx.rope_cos is not None:
            p = ctx.n_prefix
            cos, sin = ctx.rope_cos, ctx.rope_sin                                  # (L|1, H|1, N, D)
            li = self.layer_idx if cos.shape[0] > 1 else 0
            cos, sin = cos[li].to(q.dtype), sin[li].to(q.dtype)
            qp, kp = q[:, :, p:], k[:, :, p:]
            if ctx.pre_rot_fn is not None:
                qp, kp = ctx.pre_rot_fn(qp, kp, self.layer_idx)
            qp = apply_rotary(qp, cos, sin)
            kp = apply_rotary(kp, cos, sin)
            g = None
            if ctx.gate_fn is not None:                                            # per-layer gates from tokens
                g = ctx.gate_fn(x[:, p:], self.layer_idx).to(q.dtype)             # (B, H, N, nb)
            elif ctx.rope_gate is not None:
                g = ctx.rope_gate                                                  # (B, L|1, H, N, nb)
                g = g[:, self.layer_idx] if g.shape[1] > 1 else g[:, 0]           # (B, H, N, nb)
            if g is not None:
                nb = g.shape[-1]
                Bq, Hq, Nq, D = qp.shape
                # rotate-half layout: [x-pairs | y-pairs | x-pairs | y-pairs], bands contiguous per axis
                gv = g[..., None, None, :, None]                                  # (B, H, N, 1, 1, nb, 1)
                qp = (qp.view(Bq, Hq, Nq, 2, 2, nb, D // (4 * nb)) * gv).view(Bq, Hq, Nq, D)
                if ctx.rope_gate_k:
                    kp = (kp.view(Bq, Hq, Nq, 2, 2, nb, D // (4 * nb)) * gv).view(Bq, Hq, Nq, D)
            q = torch.cat([q[:, :, :p], qp], dim=2)
            k = torch.cat([k[:, :, :p], kp], dim=2)

        if ctx is not None and ctx.qk_fn is not None:
            q, k = ctx.qk_fn(q, k, self.layer_idx)
        if ctx is not None and ctx.xqk_fn is not None:                            # content-dependent rotations
            q, k = ctx.xqk_fn(x, q, k, self.layer_idx)                            # (CARoPE, pe_extra)

        bias = ctx.attn_bias if ctx is not None else None
        if ctx is not None and ctx.bias_fn is not None:
            b2 = ctx.bias_fn(x, self.layer_idx)
            bias = b2 if bias is None else bias + b2
        keep = ctx is not None and ctx.extra.get("keep_attn", False)
        attn_fn = ctx.attn_fn if ctx is not None else None                       # iRPE / GPSA / WaveletRPE
        if bias is None and not keep and attn_fn is None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if bias is not None:
                attn = attn + bias
            attn = attn.softmax(dim=-1) if attn_fn is None else attn_fn(attn, q, self.layer_idx)
            if keep:
                self.last_attn = attn.detach()
            attn = self.attn_drop(attn)
            out = attn @ v

        if ctx is not None and ctx.value_fn is not None:                          # WaveLePE (ViPER)
            p = ctx.n_prefix
            out = torch.cat([out[:, :, :p], out[:, :, p:] + ctx.value_fn(v[:, :, p:], self.layer_idx)], dim=2)

        if ctx is not None and ctx.lepe:
            p = ctx.n_prefix
            vg = v[:, :, p:].transpose(1, 2).reshape(B, N - p, C).transpose(1, 2).reshape(B, C, *ctx.grid)
            lepe = self.lepe_conv(vg).flatten(2).transpose(1, 2)                 # (B, N-p, C)
            lepe = lepe.reshape(B, N - p, self.num_heads, self.head_dim).transpose(1, 2)
            out = torch.cat([out[:, :, :p], out[:, :, p:] + lepe], dim=2)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(self.norm(out)))
