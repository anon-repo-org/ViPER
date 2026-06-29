"""
viper.model
─────────────────────────────────────────────────────────────────
DeiT-Tiny wrapper with swappable positional encoding.

DeiTWithCustomPE:
  - Loads a timm DeiT-Tiny (pretrained by default; from-scratch optional)
  - Zeroes out and freezes the backbone's built-in absolute PE so the
    candidate PE module provides all positional information
  - Supports input-only PEs (none, learned, sincos2d, relative2d, cpe,
    viper) and per-block PEs (multipeg)

For attention-internal PEs (rope_mixed, alibi2d) use this class with
pe_type='none' and then call apply_internal_pe_to_deit (see extra_pes.py).

For multi-resolution training, make_pe_dynamic patches the PE modules so
they handle dynamic input resolutions in the forward pass.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("timm is required: pip install timm")

from .viper import ViPERConfig, ViPERFeatureExtractor
from .pe_methods import (
    NoPE, LearnedPE, SinCos2DPE, Relative2DPE, CPE, PerBlockPEG,
)


# =============================================================================
# DeiT wrapper with swappable PE
# =============================================================================

class DeiTWithCustomPE(nn.Module):
    """Wraps timm DeiT-Tiny with a swappable PE module.

    The backbone's built-in pos_embed is zero-initialized and frozen so
    all positional information comes from the candidate PE.

    Modes:
      - Input-only PEs (none, learned, sincos2d, relative2d, cpe, viper):
        the PE adjustment is added to patch+CLS embeddings before block 0.
      - Per-block PEs (multipeg): the model holds a ModuleList of
        PerBlockPEG modules and applies one before each transformer block.
    """
    def __init__(self, num_classes: int, image_size: int, patch_size: int,
                 pe_type: str, viper_cfg: Optional[ViPERConfig] = None,
                 num_blocks_for_peg: int = 12, pretrained: bool = True):
        super().__init__()
        self.pe_type   = pe_type
        self.viper_cfg = viper_cfg
        self.image_size = image_size
        self.patch_size = patch_size

        # Load pretrained DeiT-Tiny
        self.backbone = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=image_size,
        )

        d_model = self.backbone.embed_dim                    # 192
        H_p = image_size // patch_size
        W_p = image_size // patch_size
        num_patches = H_p * W_p
        self.H_p, self.W_p = H_p, W_p
        self.d_model    = d_model
        self.num_blocks = len(self.backbone.blocks)

        # Zero out and freeze the backbone's built-in pos_embed.
        with torch.no_grad():
            self.backbone.pos_embed.zero_()
        self.backbone.pos_embed.requires_grad = False

        # Build the candidate PE module
        if pe_type == "viper":
            assert viper_cfg is not None, "viper_cfg required when pe_type='viper'"
            self.pe_module = ViPERFeatureExtractor(
                viper_cfg, in_channels=3,
                image_h=image_size, image_w=image_size,
                patch_size=patch_size, d_model=d_model,
            )
            self.per_block = None
        elif pe_type == "none":
            self.pe_module = NoPE(num_patches + 1, d_model)
            self.per_block = None
        elif pe_type == "learned":
            self.pe_module = LearnedPE(num_patches + 1, d_model)
            self.per_block = None
        elif pe_type == "sincos2d":
            self.pe_module = SinCos2DPE(H_p, W_p, d_model)
            self.per_block = None
        elif pe_type == "relative2d":
            self.pe_module = Relative2DPE(H_p, W_p, d_model)
            self.per_block = None
        elif pe_type == "cpe":
            self.pe_module = CPE(H_p, W_p, d_model)
            self.per_block = None
        elif pe_type == "multipeg":
            self.pe_module = None
            self.per_block = nn.ModuleList([
                PerBlockPEG(H_p, W_p, d_model) for _ in range(self.num_blocks)
            ])
        else:
            raise ValueError(f"Unknown pe_type: {pe_type}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]

        # Patch embed + CLS
        x = self.backbone.patch_embed(image)                 # (B, N, d)
        cls_tok = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tok, x], dim=1)                   # (B, N+1, d)

        # Apply input-only PE
        if self.pe_module is not None:
            if self.pe_type == "cpe":
                x = self.pe_module(x)                         # CPE returns the modified sequence
            elif self.pe_type == "viper":
                x = x + self.pe_module(image)
            else:
                x = x + self.pe_module(x)

        # DeiT's pos_drop
        x = self.backbone.pos_drop(x)

        # Transformer blocks (with optional per-block PEG)
        for i, block in enumerate(self.backbone.blocks):
            if self.per_block is not None:
                x = self.per_block[i](x)
            x = block(x)

        # Norm + classifier head
        x = self.backbone.norm(x)
        return self.backbone.head(x[:, 0])


# =============================================================================
# Multi-resolution patching
# =============================================================================

def make_pe_dynamic(model: DeiTWithCustomPE):
    """Patch ViPER, CPE, Multi-PEG, and No-PE to handle dynamic input sizes.

    Used by multi-resolution training: the model is constructed for one
    "base" resolution, then patched so that its PE modules read H,W from
    the actual input tensor at forward time. Supported pe_types:
    viper, cpe, multipeg, none.
    """
    pe_type = model.pe_type

    # Disable timm PatchEmbed's input-size assertion.
    pe = model.backbone.patch_embed
    pe.strict_img_size = False
    if hasattr(pe, "dynamic_img_pad"):
        pe.dynamic_img_pad = True

    if pe_type == "viper":
        ve = model.pe_module

        def viper_forward(image: torch.Tensor) -> torch.Tensor:
            B, _, H, W = image.shape
            H_p = H // model.patch_size
            W_p = W // model.patch_size

            if ve.cfg.use_channel_proj:
                x = ve.channel_proj(image)
            else:
                x = image.mean(dim=1, keepdim=True)

            Yl, Yh = ve.dwt(x)

            gate = ve._compute_gate(0).view(1, ve.d_pe, 1, 1)
            ll = Yl * gate
            if ll.shape[-1] != W_p or ll.shape[-2] != H_p:
                ll = F.adaptive_avg_pool2d(ll, (H_p, W_p))
            acc = ll

            for j in range(ve.cfg.n_levels):
                details = Yh[j]
                for d in range(3):
                    sb_idx = 1 + j * 3 + d
                    sb = details[:, :, d]
                    g = ve._compute_gate(sb_idx).view(1, ve.d_pe, 1, 1)
                    sb = sb * g
                    if sb.shape[-1] != W_p or sb.shape[-2] != H_p:
                        sb = F.adaptive_avg_pool2d(sb, (H_p, W_p))
                    acc = acc + sb

            pe_seq = acc.permute(0, 2, 3, 1).reshape(B, -1, ve.d_pe)
            pe_seq = ve.proj_to_dmodel(pe_seq)
            cls = ve.cls_pe.expand(B, -1, -1)
            return torch.cat([cls, pe_seq], dim=1)

        ve.forward = viper_forward

    elif pe_type == "cpe":
        cpe = model.pe_module

        def cpe_forward(x_tokens: torch.Tensor) -> torch.Tensor:
            cls, pts = x_tokens[:, :1], x_tokens[:, 1:]
            B, N, C = pts.shape
            side = int(math.sqrt(N))
            sp = pts.transpose(1, 2).reshape(B, C, side, side)
            sp = cpe.proj(sp) + sp
            pts = sp.flatten(2).transpose(1, 2)
            return torch.cat([cls + cpe.cls_pe.expand(B, -1, -1), pts], dim=1)

        cpe.forward = cpe_forward

    elif pe_type == "multipeg":
        for peg in model.per_block:
            def make_peg_forward(peg_mod):
                def peg_forward(x_tokens):
                    cls, pts = x_tokens[:, :1], x_tokens[:, 1:]
                    B, N, C = pts.shape
                    side = int(math.sqrt(N))
                    sp = pts.transpose(1, 2).reshape(B, C, side, side)
                    sp = peg_mod.conv(sp) + sp
                    pts = sp.flatten(2).transpose(1, 2)
                    return torch.cat([cls, pts], dim=1)
                return peg_forward
            peg.forward = make_peg_forward(peg)

    elif pe_type == "none":
        nope = model.pe_module

        def nope_forward(x, image=None):
            return torch.zeros(x.shape[0], x.shape[1], nope.d_model,
                                device=x.device, dtype=x.dtype)

        nope.forward = nope_forward

    else:
        raise ValueError(
            f"pe_type={pe_type} doesn't natively support multi-resolution. "
            f"Use one of: viper, cpe, multipeg, none."
        )
