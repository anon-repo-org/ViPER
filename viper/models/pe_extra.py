"""Recent positional-encoding baselines, re-implemented from the official code / papers
(full spec with sources: benchmark/baselines.md). All plug into PEAttention through two
PEContext hooks:

  ctx.qk_fn(q, k, layer)  -> (q, k)       rotations applied to the full sequence (method
                                           decides what happens to the CLS token)
  ctx.bias_fn(x, layer)   -> (B|1, H, N, N) additive logit bias (x = attention input, i.e. the
                                           pre-attention LayerNorm output)
  ctx.xqk_fn(x, q, k, layer) -> (q, k)    content-dependent rotations from the attention input (CARoPE)
  ctx.attn_fn(logits, q, layer) -> (B, H, N, N) attention probabilities replacing the softmax
                                           (logits = q.k*d^-0.5 (+ bias); q = unscaled full-sequence
                                           query). Used by iRPE / wavelet RPE (query-dependent bias
                                           q_i.r_ij) and ConViT GPSA (gated mixture of two softmaxes).

Choices for points the sources leave open are documented per class.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pe import LearnedAPE, PEContext, grid_coords


def _fp32():
    """generators, angles, matrix exponentials and biases must not run in bf16 (autocast would
    cast einsum/matmul inputs): bf16 angles carry ~0.1 rad error and matrix_exp diverges."""
    return torch.autocast("cuda", enabled=False)


def _rot_pairs(x, cos, sin):
    """rotate adjacent channel pairs (2c, 2c+1) — layout used by LieRE/STRING/Spiral sketches."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


# --------------------------------------------------------------------------- LieRE
class LieRE(nn.Module):
    """LieRE (ICML'25). Per layer and head, g×g skew-symmetric generators per axis
    (strict upper triangle ~ U(0, 2π)); R_i = blockdiag exp(Σ_a p_ia A_a) applied to q and k.
    Coordinates linspace(0, n/n_train, n) per axis (official code; proportional extrapolation).
    CLS -> identity. g = 8 (paper's headline LieRE8)."""
    kind = "attention"

    def __init__(self, head_dim, num_heads, depth, grid, g: int = 8):
        super().__init__()
        self.g, self.train_grid = g, grid
        self.raw = nn.Parameter(torch.rand(depth, 2, num_heads, head_dim // g, g, g) * 2 * math.pi)

    def coords(self, grid, device, pos_scale):
        ys = torch.linspace(0, grid[0] / self.train_grid[0], grid[0], device=device)
        xs = torch.linspace(0, grid[1] / self.train_grid[1], grid[1], device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([yy.reshape(-1), xx.reshape(-1)], -1) * pos_scale

    def forward(self, tokens, ctx: PEContext):
        c = self.coords(ctx.grid, tokens.device, ctx.extra.get("pos_scale", 1.0))
        p = ctx.n_prefix

        def qk(q, k, layer):
            with _fp32():
                U = torch.triu(self.raw[layer].float(), 1)
                A = U - U.transpose(-1, -2)                                        # (2,H,G,g,g)
                M = torch.einsum("na,ahgij->hngij", c.float(), A).contiguous()     # (H,N,G,g,g)
                R = torch.linalg.matrix_exp(M).to(q.dtype)

            def apply(x):
                xp = x[:, :, p:]
                xs = xp.reshape(*xp.shape[:3], -1, self.g, 1)
                return torch.cat([x[:, :, :p], (R @ xs).reshape(xp.shape)], 2)
            return apply(q), apply(k)
        ctx.qk_fn = qk
        return tokens


# --------------------------------------------------------------------------- ComRoPE
class ComRoPE(nn.Module):
    """ComRoPE (CVPR'25), AP or LD variant, block size b = 8 (paper), per layer and head,
    P ~ N(0, init_std=1) (code default), no APE. Coordinates (idx + 0.5) / n of the current
    grid and CLS at (0.5, 0.5) (official code), rotated like a centre patch."""
    kind = "attention"

    def __init__(self, head_dim, num_heads, depth, b: int = 8, variant: str = "LD", init_std: float = 1.0):
        super().__init__()
        G = head_dim // b
        self.b, self.variant = b, variant
        self.P = nn.Parameter(torch.randn(depth, num_heads, 1, G, b, b) * init_std)
        if variant == "LD":
            self.theta = nn.Parameter(torch.randn(depth, 1, 2, G, 1, 1))
        else:
            m = torch.zeros(2, G)
            for a in range(2):
                m[a, a::2] = 1
            self.register_buffer("mask", m.view(1, 2, G, 1, 1), persistent=False)

    def forward(self, tokens, ctx: PEContext):
        h, w = ctx.grid
        s = ctx.extra.get("pos_scale", 1.0)
        c = grid_coords(ctx.grid, tokens.device)
        c = (c + 0.5) / torch.tensor([h, w], device=tokens.device, dtype=c.dtype)
        c = torch.cat([torch.full((ctx.n_prefix, 2), 0.5, device=c.device), c]) * s     # CLS at centre

        def qk(q, k, layer):
            with _fp32():
                P = self.P[layer].float()
                S = P - P.transpose(-1, -2)                                        # (H,1,G,b,b)
                A = S * (self.theta[layer].float() if self.variant == "LD" else self.mask)   # (H,2,G,b,b)
                M = torch.einsum("na,hagij->hngij", c.float(), A).contiguous()
                R = torch.linalg.matrix_exp(M).to(q.dtype)                         # (H,N,G,b,b)

            def apply(x):
                return (R @ x.reshape(*x.shape[:3], -1, self.b, 1)).reshape(x.shape)
            return apply(q), apply(k)
        ctx.qk_fn = qk
        return tokens


# --------------------------------------------------------------------------- STRING (Cayley)
class CayleySTRING(nn.Module):
    """Cayley-STRING (ICML'25): R(r) z = RoPE-Mixed(r) (P z), P = (I - S)(I + S)^{-1}, S learned
    skew-symmetric per layer and head, RoPE-Mixed frequencies learned (max-wavelength-100 init
    with a random per-head axis angle, as in RoPE-ViT). No official code; S = 0 init (start
    exactly at RoPE-Mixed), integer patch coordinates, CLS untouched, no APE (assumptions)."""
    kind = "attention"

    def __init__(self, head_dim, num_heads, depth, base: float = 100.0):
        super().__init__()
        self.d = head_dim
        self.S_raw = nn.Parameter(torch.zeros(depth, num_heads, head_dim, head_dim))
        mag = 1.0 / base ** (torch.arange(0, head_dim, 4).float() / head_dim)          # (d/4,)
        fr = []
        for _ in range(depth):
            ang = torch.rand(num_heads) * 2 * math.pi
            fx = torch.cat([mag[None] * ang.cos()[:, None], mag[None] * (ang + math.pi / 2).cos()[:, None]], 1)
            fy = torch.cat([mag[None] * ang.sin()[:, None], mag[None] * (ang + math.pi / 2).sin()[:, None]], 1)
            fr.append(torch.stack([fy, fx], 0))                                        # (2,H,d/2)
        self.freqs = nn.Parameter(torch.stack(fr))                                     # (L,2,H,d/2)

    def forward(self, tokens, ctx: PEContext):
        c = grid_coords(ctx.grid, tokens.device, scale=ctx.extra.get("pos_scale", 1.0))
        p = ctx.n_prefix
        I = torch.eye(self.d, device=tokens.device)

        def qk(q, k, layer):
            with _fp32():
                Sr = self.S_raw[layer].float()
                S = Sr - Sr.transpose(-1, -2)
                P = torch.linalg.solve(I + S, I - S).to(q.dtype)                       # (H,d,d)
                ang = torch.einsum("na,ahf->hnf", c.float(), self.freqs[layer].float())
                cos, sin = ang.cos().to(q.dtype), ang.sin().to(q.dtype)

            def apply(x):
                xp = torch.einsum("hij,bhnj->bhni", P, x[:, :, p:])
                return torch.cat([x[:, :, :p], _rot_pairs(xp, cos, sin)], 2)
            return apply(q), apply(k)
        ctx.qk_fn = qk
        return tokens


# --------------------------------------------------------------------------- Spiral RoPE
class SpiralRoPE(nn.Module):
    """Spiral RoPE (ICML'26): fixed 2D RoPE over K = 16 equally spaced directions (paper setting
    for DeiT), frequencies 1.5·10000^{-t/(d/4)} paired to directions as in Eq. 13, shared over
    heads/layers, patch tokens only, combined with a learned APE (as in the paper). Uses the
    paper's exact projection p·u_k (not the official grid_sample border clamp)."""
    kind = "attention"

    def __init__(self, dim, head_dim, grid, K: int = 16, theta: float = 10000.0, scale: float = 1.5):
        super().__init__()
        nf, R = head_dim // 4, K // 2
        assert head_dim % (2 * K) == 0
        w = scale * theta ** (-torch.arange(nf).float() / nf)
        d_idx = (torch.arange(nf) // 2) % R
        self.register_buffer("w", torch.cat([w, w]), persistent=False)
        self.register_buffer("dir", torch.cat([d_idx, d_idx + R]), persistent=False)
        u = torch.stack([(torch.arange(K) * math.pi / K).cos(), (torch.arange(K) * math.pi / K).sin()], -1)
        self.register_buffer("u", u, persistent=False)
        self.ape = LearnedAPE(dim, grid)

    def forward(self, tokens, ctx: PEContext):
        tokens = self.ape(tokens, ctx)
        c = grid_coords(ctx.grid, tokens.device, scale=ctx.extra.get("pos_scale", 1.0))
        with _fp32():
            ang = (c.float() @ self.u.T.float())[:, self.dir] * self.w                 # (N, d/2)
            cos, sin = ang.cos(), ang.sin()
        p = ctx.n_prefix

        def qk(q, k, layer):
            cs, sn = cos.to(q.dtype), sin.to(q.dtype)

            def apply(x):
                return torch.cat([x[:, :, :p], _rot_pairs(x[:, :, p:], cs, sn)], 2)
            return apply(q), apply(k)
        ctx.qk_fn = qk
        return tokens


# --------------------------------------------------------------------------- LookHere
class LookHere(nn.Module):
    """LookHere (NeurIPS'24), LH-90 FOVs. With 6 heads (DeiT-S) we use the official code's only
    working 6-head layout: 4 directed (quadrant) heads + 2 undirected heads (slopes 1/2, 1/8).
    bias = -s_g s_l(l) s_h(h) ||Δ||_2 inside the FOV, -inf outside; s_l = linspace(1.5, 0.5, L),
    s_g = 2; CLS row/column 0. No parameters, no APE."""
    kind = "attention"

    def __init__(self, num_heads, depth, variant: str = "LH-90", s_g: float = 2.0):
        super().__init__()
        self.H, self.L, self.variant, self.s_g = num_heads, depth, variant, s_g
        self._cache = {}

    def bias(self, grid, n_prefix, device):
        key = (grid, n_prefix, str(device))
        if key in self._cache:
            return self._cache[key]
        c = grid_coords(grid, device)
        dy = c[None, :, 0] - c[:, None, 0]
        dx = c[None, :, 1] - c[:, None, 1]
        dist = (dy ** 2 + dx ** 2).sqrt()
        up, down, left, right = dy <= 0, dy >= 0, dx <= 0, dx >= 0
        cu, cd, cl, cr = dy <= -dx.abs(), dy >= dx.abs(), dx <= -dy.abs(), dx >= dy.abs()
        fov = {"LH-180": [up, down, left, right, cu | cr, cu | cl, cd | cr, cd | cl],
               "LH-90": [up & left, up & right, down & left, down & right, cu, cd, cl, cr],
               "LH-45": [cu & left, cd & left, cl & up, cr & up, cu & right, cd & right, cl & down, cr & down]}[self.variant]
        undirected = [0.25, 0.0625, 0.015625, 0.00390625]
        if self.H == 12:
            fovs, sh = fov, [0.5] * 8 + undirected
        elif self.H == 6:
            fovs, sh = fov[:4], [0.5] * 4 + undirected[:2]
        elif self.H == 3:
            fovs, sh = fov[:2], [0.5] * 2 + undirected[:1]
        else:
            raise ValueError(f"LookHere: no head layout for {self.H} heads")
        n, P = dist.shape[0], n_prefix
        neg = -1e4                                                                    # exp(-1e4) = 0 in fp32/bf16
        B = torch.zeros(self.L, self.H, n + P, n + P, device=device)
        for l, sl in enumerate(torch.linspace(1.5, 0.5, self.L).tolist()):
            for h in range(self.H):
                b = -self.s_g * sl * sh[h] * dist
                if h < len(fovs):
                    b = torch.where(fovs[h], b, torch.full_like(b, neg))
                B[l, h, P:, P:] = b
        self._cache = {key: B}
        return B

    def forward(self, tokens, ctx: PEContext):
        B = self.bias(ctx.grid, ctx.n_prefix, tokens.device)
        ctx.bias_fn = lambda x, layer: B[layer][None].to(x.dtype)
        return tokens


# --------------------------------------------------------------------------- PaPE
class PaPE(nn.Module):
    """PaPE (2026): per layer, M = 50 parabolas per head over projected positions s = W_p r,
    content-dependent curvature a = softplus(W_a x) and slope b = W_b x from the pre-attention
    LayerNorm output; logits = (q·k - Σ a Δs² - Σ b Δs) / sqrt(d). Positions are integer
    (col,row) 1..n, CLS at (0,0) (official code). No APE."""
    kind = "attention"

    def __init__(self, dim, num_heads, depth, head_dim, M: int = 50):
        super().__init__()
        self.H, self.M, self.scale = num_heads, M, head_dim ** -0.5
        self.position = nn.ModuleList([nn.Linear(2, num_heads * M, bias=False) for _ in range(depth)])
        self.ab = nn.ModuleList([nn.Linear(dim, 2 * num_heads * M, bias=False) for _ in range(depth)])

    def forward(self, tokens, ctx: PEContext):
        c = grid_coords(ctx.grid, tokens.device, scale=ctx.extra.get("pos_scale", 1.0)).flip(-1) + 1
        r = torch.cat([torch.zeros(ctx.n_prefix, 2, device=c.device), c])               # (N,2), CLS at 0

        def bias(x, layer):
            B, N, _ = x.shape
            a, b = self.ab[layer](x).view(B, N, 2, self.H, self.M).permute(2, 0, 3, 1, 4)      # (B,H,N,M)
            with _fp32():
                s = self.position[layer](r.float()).view(N, self.H, self.M).permute(1, 0, 2)   # (H,N,M)
                a = F.softplus(a.float()); b = b.float()
                sT, s2T = s.transpose(-1, -2)[None], (s ** 2).transpose(-1, -2)[None]
                quad = a @ s2T - 2 * (a * s[None]) @ sT + (a * (s ** 2)[None]).sum(-1, keepdim=True)
                lin = b @ sT - (b * s[None]).sum(-1, keepdim=True)
                return (-(quad + lin) * self.scale).to(x.dtype)
        ctx.bias_fn = bias
        return tokens


# --------------------------------------------------------------------------- iRPE
@torch.no_grad()
def _irpe_piecewise_index(rp, alpha, beta, gamma, dtype=torch.long):
    """iRPE piecewise index g(x), Eq. (18); verbatim port of Cream/iRPE `irpe.py::piecewise_index`:
    |x| <= alpha -> round(x); else sign(x) * min(beta, round(alpha + ln(|x|/alpha)/ln(gamma/alpha)
    * (beta - alpha))), then cast (the float clip at beta=3.8 truncates to 3). Output in [-int(beta), int(beta)]."""
    rp_abs = rp.abs()
    not_mask = ~(rp_abs <= alpha)
    rp_out, rp_abs_out = rp[not_mask], rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (alpha + torch.log(rp_abs_out / alpha) / math.log(gamma / alpha)
                                   * (beta - alpha)).round().clip(max=beta)).to(dtype)
    idx = rp.clone()
    if idx.dtype in (torch.float32, torch.float64):
        idx = idx.round().to(dtype)
    idx[not_mask] = y_out
    return idx


class IRPE(nn.Module):
    """iRPE (Wu et al., ICCV'21), the configuration of the released DeiT models
    (`rpe_models.py::deit_small_patch16_224_ctx_product_50_shared_k`): contextual mode, product
    method, piecewise index with ratio 1.9 (alpha, beta, gamma = 1.9, 3.8, 15.2), on keys only,
    shared over heads, one table per layer (every RPEAttention builds its own iRPE):

        logit_ij = (q_i d^-0.5) . k_j  +  (q_i d^-0.5) . r_{bucket(p_i - p_j)}

    bucket = (g(Δrow) + 3) * 7 + (g(Δcol) + 3) in [0, 49), plus one extra 'skip' bucket (id 49)
    shared by every CLS row and column (official `get_bucket_ids_2d`, skip=1) -> 50 buckets.
    Tables r are zero-initialised (official default initializer). g clips at |g| = 3, so the
    buckets of any test grid are well defined (no resampling needed). The released DeiT-iRPE
    models keep DeiT's learned APE (`rpe_vision_transformer.py` adds `pos_embed`), so `ape=True`
    (bicubic-resampled at unseen grids, as for the plain DeiT baseline)."""
    kind = "attention"

    def __init__(self, dim, head_dim, num_heads, depth, grid, ratio: float = 1.9, ape: bool = True):
        super().__init__()
        self.alpha, self.beta, self.gamma = 1 * ratio, 2 * ratio, 8 * ratio
        self.beta_int = int(self.beta)
        self.S = 2 * self.beta_int + 1
        self.nb = self.S ** 2 + 1                                                       # + skip (CLS) bucket
        self.scale = head_dim ** -0.5
        self.weight = nn.Parameter(torch.zeros(depth, head_dim, self.nb))             # shared_head=True
        self.ape = LearnedAPE(dim, grid) if ape else None
        self._cache = {}

    def buckets(self, grid, n_prefix, device, pos_scale=1.0):
        key = (grid, n_prefix, str(device), pos_scale)
        if key not in self._cache:
            c = grid_coords(grid, device, scale=pos_scale)
            if pos_scale == 1.0:
                c = c.long()
            diff = c[:, None, :] - c[None, :, :]                                         # query - key (official)
            kw = dict(alpha=self.alpha, beta=self.beta, gamma=self.gamma)
            ids = (_irpe_piecewise_index(diff[..., 0], **kw) + self.beta_int) * self.S \
                + _irpe_piecewise_index(diff[..., 1], **kw) + self.beta_int
            n, P = ids.shape[0], n_prefix
            full = torch.full((n + P, n + P), self.nb - 1, dtype=torch.long, device=device)
            full[P:, P:] = ids
            self._cache = {key: full}
        return self._cache[key]

    def forward(self, tokens, ctx: PEContext):
        if self.ape is not None:
            tokens = self.ape(tokens, ctx)
        idx = self.buckets(ctx.grid, ctx.n_prefix, tokens.device, ctx.extra.get("pos_scale", 1.0))

        def attn(logits, q, layer):
            B, H, N, _ = logits.shape
            lut = (q * self.scale) @ self.weight[layer].to(q.dtype)                    # (B,H,N,nb)
            return (logits + lut.gather(-1, idx.expand(B, H, N, N))).softmax(dim=-1)
        ctx.attn_fn = attn
        return tokens


# --------------------------------------------------------------------------- CrossFormer DPB
class _DynamicPosBias(nn.Module):
    """verbatim CrossFormer `models/crossformer.py::DynamicPosBias` (pos_dim = dim // 4)."""

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.pos_dim))
        self.pos2 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.pos_dim))
        self.pos3 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, num_heads))

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            return self.pos3(pos)
        return self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))


class DPB(nn.Module):
    """CrossFormer Dynamic Position Bias (ICLR'22). Per layer an MLP maps the integer offset
    (Δrow, Δcol) = p_i - p_j (query - key, official `relative_coords`) to one bias per head; it is
    evaluated on the (2h-1)(2w-1) offsets of the current grid and gathered (official
    `Attention.forward`), so any grid works. As in the official Attention: DynamicPosBias(dim // 4,
    heads, residual=False) -> hidden width dim // 16 (= 24 for DeiT-S), trunc_normal(0.02) Linear
    weights, zero biases (CrossFormer `_init_weights`). CrossFormer has no CLS token: CLS row and
    column get bias 0 (assumption). No APE (CrossFormer uses none)."""
    kind = "attention"

    def __init__(self, dim, num_heads, depth, residual: bool = False):
        super().__init__()
        self.pos = nn.ModuleList(_DynamicPosBias(dim // 4, num_heads, residual) for _ in range(depth))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, tokens, ctx: PEContext):
        h, w = ctx.grid
        s, P, dev = ctx.extra.get("pos_scale", 1.0), ctx.n_prefix, tokens.device
        dy, dx = torch.meshgrid(torch.arange(1 - h, h, device=dev), torch.arange(1 - w, w, device=dev), indexing="ij")
        off = torch.stack([dy.reshape(-1), dx.reshape(-1)], -1).float() * s             # ((2h-1)(2w-1), 2)
        c = grid_coords(ctx.grid, dev).long()
        rel = c[:, None, :] - c[None, :, :] + torch.tensor([h - 1, w - 1], device=dev)
        idx = rel[..., 0] * (2 * w - 1) + rel[..., 1]                                     # (n, n)

        def bias(x, layer):
            with _fp32():
                b = self.pos[layer](off)[idx].permute(2, 0, 1)                            # (H, n, n)
            return F.pad(b, (P, 0, P, 0))[None].to(x.dtype)
        ctx.bias_fn = bias
        return tokens


# --------------------------------------------------------------------------- ConViT GPSA
class GPSA(nn.Module):
    """ConViT gated positional self-attention (ICML'21, `convit.py::GPSA`). In the first
    `local_up_to_layer` = 10 blocks (ConViT default):

        A = (1 - σ(λ_h)) softmax(q k^T d^-0.5) + σ(λ_h) softmax(v_pos_h . r_ij),
        r_ij = (Δx, Δy, Δx² + Δy²), Δ = p_key - p_query (official `get_rel_indices`),

    pos_proj = Linear(3, H) (with bias) per layer, λ_h init 1. Locality init (official
    `local_init`, locality_strength 1): kernel = int(sqrt(H)); for the first kernel² heads
    v_pos = (2(h2-c), 2(h1-c), -1), i.e. -|Δ - c_h|² + const; with H = 6 heads (DeiT-S) kernel = 2,
    so heads 0-3 attend to the 4 offsets (±0.5, ±0.5) and heads 4-5 keep the trunc_normal(0.02)
    init (exactly what the official code does for a non-square head count; ConViT itself uses
    4/9/16 heads). Blocks >= 10 are plain attention. Learned APE kept (ConViT use_pos_embed=True).

    Deviations: (1) ConViT concatenates the CLS token only after the GPSA blocks; our backbone has
    it from block 0, so CLS query rows use content attention only, and for patch queries the
    positional softmax runs over patch keys only (CLS key gets positional weight 0, content weight
    1 - σ). (2) GPSA's `local_init` also sets W_v = I; W_v lives in the shared timm qkv here and is
    left at its timm init. (3) The official renormalisation `attn /= attn.sum(-1)` is omitted
    (rows already sum to 1). (4) Non-square grids supported (official assumes square).
    (5) Positional scores are computed in fp32 (bf16 logits of -|Δ|² would be coarse)."""
    kind = "attention"

    def __init__(self, dim, num_heads, depth, grid, local_up_to_layer: int = 10,
                 locality_strength: float = 1.0, ape: bool = True):
        super().__init__()
        self.H, self.n_local = num_heads, min(local_up_to_layer, depth)
        self.pos_proj = nn.ModuleList(nn.Linear(3, num_heads) for _ in range(self.n_local))
        # one 1D λ per layer (as ConViT's gating_param) so the ndim<=1 rule exempts it from weight decay
        self.gating = nn.ParameterList(nn.Parameter(torch.ones(num_heads)) for _ in range(self.n_local))
        ks = int(num_heads ** 0.5)
        center = (ks - 1) / 2 if ks % 2 == 0 else ks // 2
        with torch.no_grad():
            for lin in self.pos_proj:
                nn.init.trunc_normal_(lin.weight, std=0.02)
                nn.init.zeros_(lin.bias)
                for h1 in range(ks):
                    for h2 in range(ks):
                        pos = h1 + ks * h2
                        lin.weight[pos, 2] = -1
                        lin.weight[pos, 1] = 2 * (h1 - center)
                        lin.weight[pos, 0] = 2 * (h2 - center)
                lin.weight *= locality_strength
        self.ape = LearnedAPE(dim, grid) if ape else None

    def forward(self, tokens, ctx: PEContext):
        if self.ape is not None:
            tokens = self.ape(tokens, ctx)
        c = grid_coords(ctx.grid, tokens.device, scale=ctx.extra.get("pos_scale", 1.0))
        d = c[None, :, :] - c[:, None, :]                                                 # [i, j] = p_j - p_i
        rel = torch.stack([d[..., 1], d[..., 0], (d ** 2).sum(-1)], -1)                   # (Δx, Δy, Δx²+Δy²)
        P = ctx.n_prefix

        def attn(logits, q, layer):
            content = logits.softmax(dim=-1)
            if layer >= self.n_local:
                return content
            with _fp32():
                pos = self.pos_proj[layer](rel).permute(2, 0, 1).softmax(dim=-1)          # (H, n, n)
                g = torch.sigmoid(self.gating[layer].float()).view(-1, 1, 1).expand(-1, pos.shape[1], 1)
            pos = F.pad(pos, (P, 0, P, 0))                                                # CLS: no positional term
            g = F.pad(g, (0, 0, P, 0))                                                    # CLS rows: content only
            return content * (1 - g) + pos * g
        ctx.attn_fn = attn
        return tokens


# --------------------------------------------------------------------------- Wavelet RPE (2D)
def ricker(t):
    """Ricker / Mexican-hat wavelet without amplitude normalisation (Oka et al. Eq. 12)."""
    return (1 - t ** 2) * torch.exp(-t ** 2 / 2)


class WaveletRPE(nn.Module):
    """2D port of the wavelet-based relative positional representation (Oka et al., ICLR'25,
    arXiv 2502.02004, §5.1 Eqs. 10-14). The paper (1D, causal LM) replaces the RPE embedding by
    a fixed bank of shifted/scaled Ricker wavelets of the relative position t = m - n, one
    wavelet per head channel:

        e_mn = (q_m k_n^T + q_m p_mn^T) / sqrt(d),   p_mn[c] = ψ((t - b_c) / a_c),
        ψ(t) = (1 - t²) exp(-t²/2),   (a, b) ∈ {2^0..2^(s-1)} × {0..d/s - 1}   (d pairs)

    no amplitude 1/sqrt(a), no clipping, no parameters, same bank in every head and layer
    (paper: d = 128, s = 8 scales × 16 shifts; L_train = 512).

    2D port, mode="axial" (default): half of the head channels see Δrow, half Δcol (the same
    channel split as axial 2D RoPE), each half a 1D bank ψ((Δ - b)/a), so
    q·p_ij = q_row·ψ(Δrow_ij) + q_col·ψ(Δcol_ij). Why not the Euclidean radius: the paper's
    1D offset is signed and the causal mask supplies the direction; a bidirectional 2D encoder
    has no mask, and a radial bank would make the positional term blind to direction (iRPE
    found directed > undirected for ViTs). Per axis d/2 = 32 channels = 4 scales {1, 2, 4, 8}
    (2^4 = 16 already exceeds the max offset 13 of a 14x14 grid, mirroring the paper's largest
    scale 128 ≈ L/4) × 8 shifts. The paper's shifts {0..n-1} cover the causal half-line; for a
    signed axis we use the same half-line set on both sides, b ∈ {0, 1, 2, 3} ∪ {-0, -1, -2, -3}
    (reflection-symmetric, so horizontal flips map the bank onto itself; b = ±0 duplicates one
    channel per scale). Offsets Δ = p_query - p_key in patch units, never rescaled with the grid
    (paper: "the distance of the position expression is fixed regardless of the length"), so
    every test grid works and distant pairs fall back to content-only logits.
    mode="radial" (ablation): ψ((r - b)/a) of r = ||Δ||_2 with the paper's literal 4 scales ×
    16 shifts {0..15}. Channel -> (a, b) order (unspecified in the paper) is scale-major; with
    no other transform on q it is absorbed by W_Q. CLS row/column: no positional term (the
    paper has no CLS). No APE (the paper replaces the PE)."""
    kind = "attention"

    def __init__(self, head_dim, mode: str = "axial", scales=(1, 2, 4, 8)):
        super().__init__()
        assert mode in ("axial", "radial")
        self.mode, self.scale = mode, head_dim ** -0.5
        nc = head_dim // 2 if mode == "axial" else head_dim
        ns = nc // len(scales)
        assert ns * len(scales) == nc
        if mode == "axial":
            assert ns % 2 == 0
            shifts = list(range(ns // 2)) + [-b for b in range(ns // 2)]
        else:
            shifts = list(range(ns))
        a = torch.tensor([float(x) for x in scales]).repeat_interleave(ns)                # scale-major
        b = torch.tensor([float(x) for x in shifts]).repeat(len(scales))
        self.register_buffer("a", a, persistent=False)
        self.register_buffer("b", b, persistent=False)
        self._cache = {}

    def tables(self, grid, device, pos_scale):
        """(values (V,), index (n, n)) per channel group: Ψ is evaluated on the V distinct offsets."""
        key = (grid, str(device), pos_scale)
        if key not in self._cache:
            h, w = grid
            c = grid_coords(grid, device).long()
            rel = c[:, None, :] - c[None, :, :]                                             # query - key
            if self.mode == "axial":                                                       # separable: gather
                out = []                                                                    # (n, h) / (n, w) only
                for ax, n in ((0, h), (1, w)):
                    t = torch.arange(1 - n, n, device=device).float() * pos_scale
                    idx = c[:, ax:ax + 1] - torch.arange(n, device=device)[None] + n - 1      # (N, n): Δ = p_q - p_k
                    out.append((ricker((t[:, None] - self.b) / self.a), idx))               # (2n-1, d/2)
            else:
                r2 = (rel ** 2).sum(-1)
                t = torch.arange(int(r2.max()) + 1, device=device).float().sqrt() * pos_scale
                out = [(ricker((t[:, None] - self.b) / self.a), r2)]
            self._cache = {key: out}
        return self._cache[key]

    def forward(self, tokens, ctx: PEContext):
        tabs = self.tables(ctx.grid, tokens.device, ctx.extra.get("pos_scale", 1.0))
        P = ctx.n_prefix

        def attn(logits, q, layer):
            B, H, N, _ = logits.shape
            n = N - P
            qp = q[:, :, P:] * self.scale
            qs = qp.chunk(len(tabs), dim=-1)
            b = [(qc @ psi.T.to(q.dtype)).gather(-1, idx.expand(B, H, *idx.shape))           # (B,H,n,V) -> idx
                 for qc, (psi, idx) in zip(qs, tabs)]
            if self.mode == "axial":                                                       # row term + col term
                bias = (b[0][..., :, None] + b[1][..., None, :]).flatten(-2)
            else:
                bias = b[0]
            return (logits + F.pad(bias, (P, 0, P, 0))).softmax(dim=-1)
        ctx.attn_fn = attn
        return tokens


class CARoPE2D(nn.Module):
    """CARoPE (Veisi et al., arXiv 2507.23083, "Context-aware Rotary Position Embedding") ported to 2D.

    Paper (1D, LLM): per head h a token-dependent base frequency f_h(x_t) = 1 / (softplus(x_t W)_h + 1) in (0, 1),
    W in R^{d x H}, and the phase of pair i at position m is the cumulative sum phi_i(m) = sum_{t<=m} f_h(x_t)^i
    (RoPE is the special case of a constant f = theta_1, phi_i(m) = m theta_1^i). No official code.
    2D port (choices documented here): (i) axial like RoPE-Axial: the first D/4 pairs accumulate along the row
    (x), the last D/4 along the column (y); (ii) f is computed per layer from that layer's attention input (the
    pre-attention LayerNorm output) with a per-layer W, since 'token embedding' is the hidden state in a
    transformer layer; (iii) f^i with i = 0..M-1 (M = D/4 pairs per axis) and the RoPE-ViT base theta = 100;
    (iv) "initialised as standard RoPE": W = 0 and the bias set so that f = theta^(-1/M), which gives exactly
    the RoPE-Axial angles; the sums are exclusive (phase 0 at the first token of a row/column), because an
    inclusive sum shifts every position by +1, which does not cancel against the unrotated CLS token; (v) the CLS token is not
    rotated; cumulative sums run in fp32. The encoding is no longer strictly relative: the phase difference
    between two tokens depends on the content of the tokens between them, which is the point of the method."""
    kind = "attention"

    def __init__(self, dim: int, head_dim: int, num_heads: int, depth: int, theta: float = 100.0, lepe: bool = False):
        super().__init__()
        assert head_dim % 4 == 0
        self.M, self.num_heads = head_dim // 4, num_heads
        f0 = theta ** (-1.0 / self.M)
        b0 = math.log(math.expm1(1.0 / f0 - 1.0))                         # softplus(b0) = 1/f0 - 1
        self.W = nn.Parameter(torch.zeros(depth, dim, num_heads))
        self.b = nn.Parameter(torch.full((depth, num_heads), b0))
        self.lepe = lepe

    def forward(self, tokens, ctx: PEContext):
        h, w = ctx.grid
        P = ctx.n_prefix
        expo = torch.arange(self.M, device=tokens.device, dtype=torch.float32)

        def xqk(x, q, k, layer):
            B, H, N, D = q.shape
            with _fp32():
                z = x[:, P:].float() @ self.W[layer].float() + self.b[layer].float()      # (B, n, H)
                logf = -torch.log1p(F.softplus(z))                                         # log f, f = 1/(softplus+1)
                fi = torch.exp(logf[..., None] * expo)                                     # (B, n, H, M) = f^i
                fi = fi.view(B, h, w, H, self.M)
                ph_x = fi.cumsum(2) - fi                                                  # along the row, exclusive
                ph_y = fi.cumsum(1) - fi                                                  # along the column, exclusive
                ang = torch.cat([ph_x, ph_y], -1).flatten(1, 2).permute(0, 2, 1, 3)       # (B, H, n, D/2)
                ang = torch.cat([ang, ang], -1)
                cos, sin = ang.cos().to(q.dtype), ang.sin().to(q.dtype)

            def rot(t):
                tp = t[:, :, P:]
                x1, x2 = tp.chunk(2, dim=-1)
                return torch.cat([t[:, :, :P], tp * cos + torch.cat([-x2, x1], -1) * sin], dim=2)
            return rot(q), rot(k)
        ctx.xqk_fn = xqk
        ctx.lepe = self.lepe
        return tokens


def build_extra(name, *, dim, head_dim, num_heads, depth, grid, **kw):
    if name == "liere":
        return LieRE(head_dim, num_heads, depth, grid, **kw)
    if name in ("comrope_ld", "comrope_ap"):
        return ComRoPE(head_dim, num_heads, depth, variant=name[-2:].upper(), **kw)
    if name == "string_cayley":
        return CayleySTRING(head_dim, num_heads, depth, **kw)
    if name == "spiral":
        return SpiralRoPE(dim, head_dim, grid, **kw)
    if name == "lookhere":
        return LookHere(num_heads, depth, **kw)
    if name == "pape":
        return PaPE(dim, num_heads, depth, head_dim, **kw)
    if name == "irpe":
        return IRPE(dim, head_dim, num_heads, depth, grid, **kw)
    if name == "dpb":
        return DPB(dim, num_heads, depth, **kw)
    if name == "gpsa":
        return GPSA(dim, num_heads, depth, grid, **kw)
    if name == "wavelet_rpe":
        return WaveletRPE(head_dim, **kw)
    if name == "carope":
        return CARoPE2D(dim, head_dim, num_heads, depth, **kw)
    return None


EXTRA_NAMES = ["liere", "comrope_ld", "comrope_ap", "string_cayley", "spiral", "lookhere", "pape",
               "irpe", "dpb", "gpsa", "wavelet_rpe", "carope"]
