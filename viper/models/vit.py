"""timm ViT/DeiT backbone with a swappable positional encoding."""
from __future__ import annotations

from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn

from .attention import PEAttention
from .pe import PEContext, build_pe

BACKBONES = {
    # name: (timm model, patch)
    "deit_tiny": ("deit_tiny_patch16_224", 16),
    "deit_small": ("deit_small_patch16_224", 16),
    "deit_base": ("deit_base_patch16_224", 16),
    "vit_small": ("vit_small_patch16_224", 16),
    "vit_base": ("vit_base_patch16_224", 16),
}


class ViTWithPE(nn.Module):
    def __init__(self, backbone: str = "deit_small", pe: str = "viper", num_classes: int = 100,
                 img_size: int = 224, pretrained: bool = False, drop_path: float = 0.1,
                 pe_kwargs: Optional[dict] = None):
        super().__init__()
        timm_name, patch = BACKBONES[backbone]
        self.patch = patch
        self.pe_name = pe
        self.train_grid = (img_size // patch, img_size // patch)
        m = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes, img_size=img_size,
                              drop_path_rate=drop_path, dynamic_img_size=True)
        self.backbone = m
        if pe != "learned_timm":
            # all positional information must come from the PE module
            del m.pos_embed
            m.pos_embed = None
        self.ctx = PEContext(grid=self.train_grid, n_prefix=m.num_prefix_tokens)
        dim, heads, depth = m.embed_dim, m.blocks[0].attn.num_heads, len(m.blocks)
        for i, blk in enumerate(m.blocks):
            old = blk.attn
            new = PEAttention(dim, heads, qkv_bias=old.qkv.bias is not None,
                              qk_norm=not isinstance(old.q_norm, nn.Identity),
                              attn_drop=old.attn_drop.p, proj_drop=old.proj_drop.p, ctx=self.ctx, layer_idx=i)
            missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
            assert not unexpected and all(k.startswith("lepe_conv") for k in missing), (missing, unexpected)
            blk.attn = new
        self.pe = build_pe(pe, dim=dim, head_dim=dim // heads, num_heads=heads, depth=depth,
                           grid=self.train_grid, **(pe_kwargs or {}))

    # ------------------------------------------------------------------ #
    def pe_parameters(self):
        n = sum(p.numel() for p in self.pe.parameters())
        if self.pe_name == "lepe" or getattr(self.pe, "lepe", False) is True:
            n += sum(p.numel() for b in self.backbone.blocks for p in b.attn.lepe_conv.parameters())
        return n

    def tokens(self, image: torch.Tensor, pos_scale: float = 1.0) -> torch.Tensor:
        m, ctx = self.backbone, self.ctx
        B, _, H, W = image.shape
        grid = (H // self.patch, W // self.patch)
        ctx.clear()
        ctx.grid = grid
        ctx.extra["image"] = image
        ctx.extra["pos_scale"] = pos_scale
        x = m.patch_embed(image)                                   # (B, h, w, C) with dynamic_img_size
        x = x.reshape(B, -1, x.shape[-1])
        x = self.pe(x, ctx)                                        # additive PEs add here; others fill ctx
        prefix = [m.cls_token.expand(B, -1, -1)] if m.cls_token is not None else []
        if m.reg_token is not None:
            prefix.append(m.reg_token.expand(B, -1, -1))
        x = torch.cat(prefix + [x], dim=1)
        x = m.pos_drop(x)
        x = m.norm_pre(x)
        for i, blk in enumerate(m.blocks):
            x = blk(x)
            if self.pe.kind == "conv":
                x = self.pe.after_block(i, x, ctx)
        return m.norm(x)

    def forward(self, image: torch.Tensor, pos_scale: float = 1.0) -> torch.Tensor:
        x = self.tokens(image, pos_scale)
        return self.backbone.forward_head(x)


def build_model(**kw) -> ViTWithPE:
    return ViTWithPE(**kw)
