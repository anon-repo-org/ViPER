"""Positional-encoding zoo for plain ViTs (timm VisionTransformer backbones).

Every PE is one of three kinds, all driven from a single `PEContext` that the
model wrapper fills once per forward pass:

  * additive   : a (B, N, C) tensor added to the patch tokens before block 0
                 (none / learned APE / 2D sin-cos / learnable Fourier / wave_add)
  * attention  : rotary tables, attention biases or per-token gates consumed
                 inside every `PEAttention` (rope_axial / rope_mixed / alibi2d /
                 relbias / lepe / viper)
  * conv       : a PEG (depthwise conv) applied between blocks (cpe)

All coordinates are in *patch units* of the current grid, so every method that
is table-free works at any test resolution without code changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DWTForward
except ImportError:  # pragma: no cover
    DWTForward = None


# --------------------------------------------------------------------------- #
# Shared per-forward context
# --------------------------------------------------------------------------- #
@dataclass
class PEContext:
    grid: Tuple[int, int] = (14, 14)
    n_prefix: int = 1                       # CLS (and register) tokens, never rotated / biased
    rope_cos: Optional[torch.Tensor] = None  # (1|H, N, D) cos table for patch tokens
    rope_sin: Optional[torch.Tensor] = None
    rope_gate: Optional[torch.Tensor] = None  # (B, H, N, D) multiplicative gate on rotated q (ViPER)
    rope_gate_k: bool = False               # also gate k
    attn_bias: Optional[torch.Tensor] = None  # (1|B, H, N+P, N+P) additive bias on logits
    lepe: bool = False                      # CSWin LePE: v + dwconv(v) inside attention
    qk_fn: Optional[object] = None          # (q, k, layer) -> (q, k): per-layer rotations (pe_extra)
    xqk_fn: Optional[object] = None         # (x, q, k, layer) -> (q, k): rotations that depend on the layer input
    bias_fn: Optional[object] = None        # (x, layer) -> (B|1, H, N, N) logit bias (pe_extra)
    gate_fn: Optional[object] = None        # (x_patches, layer) -> (B, H, N, nb) rope band gates (ViPER feature gates)
    value_fn: Optional[object] = None       # (v_patches, layer) -> (B, H, N, D) added to the head output (WaveLePE)
    pre_rot_fn: Optional[object] = None     # (q_p, k_p, layer) -> (q_p, k_p) before the rotation (STRING basis P)
    attn_fn: Optional[object] = None        # (logits, q, layer) -> (B, H, N, N) attention probabilities; replaces
                                            # the softmax. logits = q.k*d^-0.5 (+ bias), q = full-sequence query after
                                            # rotations/qk_fn, unscaled (iRPE, ConViT GPSA, wavelet RPE in pe_extra)
    extra: Dict = field(default_factory=dict)

    def clear(self):
        self.qk_fn = self.bias_fn = self.gate_fn = self.value_fn = self.pre_rot_fn = self.attn_fn = None
        self.xqk_fn = None
        self.rope_cos = self.rope_sin = self.rope_gate = self.attn_bias = None
        self.rope_gate_k = False
        self.lepe = False
        self.extra = {}


def grid_coords(grid: Tuple[int, int], device, dtype=torch.float32, scale: float = 1.0) -> torch.Tensor:
    """(N, 2) tensor of (y, x) patch coordinates, row-major, optionally rescaled
    (scale < 1 implements positional interpolation for extrapolation studies)."""
    h, w = grid
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                            torch.arange(w, device=device, dtype=dtype), indexing="ij")
    return torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1) * scale


# --------------------------------------------------------------------------- #
# Additive PEs
# --------------------------------------------------------------------------- #
class NoPE(nn.Module):
    kind = "additive"

    def forward(self, tokens, ctx: PEContext):
        return tokens


class LearnedAPE(nn.Module):
    """Learned absolute PE (ViT/DeiT). Bicubic-resampled at unseen grids."""
    kind = "additive"

    def __init__(self, dim: int, grid: Tuple[int, int]):
        super().__init__()
        self.grid = grid
        self.pe = nn.Parameter(torch.zeros(1, grid[0] * grid[1], dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, tokens, ctx: PEContext):
        pe = self.pe
        if ctx.grid != self.grid:
            pe = pe.reshape(1, *self.grid, -1).permute(0, 3, 1, 2)
            pe = F.interpolate(pe, size=ctx.grid, mode="bicubic", align_corners=False)
            pe = pe.permute(0, 2, 3, 1).reshape(1, -1, tokens.shape[-1])
        return tokens + pe


def sincos2d(grid: Tuple[int, int], dim: int, device, temperature: float = 10000.0) -> torch.Tensor:
    """MAE / MoCo-v3 2D sin-cos table, (N, dim), in patch units (table-free)."""
    assert dim % 4 == 0
    h, w = grid
    gy, gx = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    pos_dim = dim // 4
    omega = 1.0 / (temperature ** (torch.arange(pos_dim, device=device, dtype=torch.float32) / pos_dim))
    out_y = gy.reshape(-1, 1) * omega[None]
    out_x = gx.reshape(-1, 1) * omega[None]
    return torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=1)


class SinCos2D(nn.Module):
    kind = "additive"

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, tokens, ctx: PEContext):
        return tokens + sincos2d(ctx.grid, self.dim, tokens.device).to(tokens.dtype)[None]


class LearnableFourierPE(nn.Module):
    """Learnable Fourier features (Li et al., NeurIPS 2021): MLP(cos/sin(Wx)) on
    normalised coordinates in [0,1]^2, so it is resolution-free by construction."""
    kind = "additive"

    def __init__(self, dim: int, n_freq: int = 32, hidden: int = 128, gamma: float = 1.0):
        super().__init__()
        self.Wr = nn.Linear(2, n_freq, bias=False)
        nn.init.normal_(self.Wr.weight, std=gamma ** -2)
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.n_freq = n_freq

    def forward(self, tokens, ctx: PEContext):
        h, w = ctx.grid
        c = grid_coords(ctx.grid, tokens.device) / torch.tensor([max(h - 1, 1), max(w - 1, 1)],
                                                                device=tokens.device)
        proj = self.Wr(c) * (2 * math.pi)
        f = torch.cat([proj.cos(), proj.sin()], dim=-1) / math.sqrt(self.n_freq)
        return tokens + self.mlp(f)[None].to(tokens.dtype)


# --------------------------------------------------------------------------- #
# Wavelet front-end (shared by wave_add and ViPER)
# --------------------------------------------------------------------------- #
class WaveletFrontEnd(nn.Module):
    """Grayscale -> J-level 2D DWT. Returns LL and the list of (LH, HL, HH)."""

    def __init__(self, levels: int = 3, wavelet: str = "db4"):
        super().__init__()
        assert DWTForward is not None, "pip install pytorch_wavelets"
        self.levels = levels
        self.dwt = DWTForward(J=levels, wave=wavelet, mode="symmetric")
        self.register_buffer("gray", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor):
        x = image.float()
        x = (x * self.gray).sum(1, keepdim=True) if x.shape[1] == 3 else x.mean(1, keepdim=True)
        with torch.autocast(device_type=x.device.type, enabled=False):
            ll, hs = self.dwt(x)
        return ll, hs


class FastDWT(nn.Module):
    """J-level separable 2D DWT as four stride-2 convolutions per level (filters
    from PyWavelets, reflect padding). Same subband layout as pytorch_wavelets'
    DWTForward (LL, [(LH, HL, HH) per level]) but ~10x cheaper for small
    single-channel inputs because each level is a single kernel launch."""

    def __init__(self, levels: int = 4, wavelet: str = "db4"):
        super().__init__()
        import pywt
        w = pywt.Wavelet(wavelet)
        lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)   # conv (not correlation) convention
        hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
        k = torch.stack([torch.outer(lo, lo), torch.outer(hi, lo), torch.outer(lo, hi), torch.outer(hi, hi)])
        self.register_buffer("kernel", k[:, None], persistent=False)          # (4, 1, L, L): LL, LH, HL, HH
        self.levels, self.L = levels, len(w.dec_lo)
        self.register_buffer("gray", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor):
        x = image.float()
        x = (x * self.gray).sum(1, keepdim=True) if x.shape[1] == 3 else x.mean(1, keepdim=True)
        with torch.autocast(device_type=x.device.type, enabled=False):
            hs = []
            p = self.L // 2
            for _ in range(self.levels):
                xp = F.pad(x, (p - 1, p, p - 1, p), mode="reflect")
                y = F.conv2d(xp, self.kernel, stride=2)                          # (B, 4, h/2, w/2)
                x, d = y[:, :1], y[:, 1:]
                hs.append(d[:, None])                                           # (B, 1, 3, h, w) like pytorch_wavelets
        return x, hs


def pool_to_grid(x: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == tuple(grid):
        return x
    if x.shape[-2] >= grid[0] and x.shape[-1] >= grid[1]:
        return F.adaptive_avg_pool2d(x, grid)
    return F.interpolate(x, size=grid, mode="bilinear", align_corners=False)


class WaveAdditivePE(nn.Module):
    """The *original* ViPER (WACV submission) kept as an ablation: gated, pooled
    wavelet coefficients projected to C and added to the tokens. Kept bit-for-bit
    faithful to notebooks/viper_v5.py."""
    kind = "additive"

    def __init__(self, dim: int, levels: int = 3, wavelet: str = "db4", d_pe: int = 32):
        super().__init__()
        self.fe = WaveletFrontEnd(levels, wavelet)
        self.n_sub = 1 + 3 * levels
        self.d_pe = d_pe
        self.scale_emb = nn.Parameter(torch.randn(self.n_sub, d_pe) * 0.02)
        self.W_g = nn.Linear(d_pe, d_pe, bias=False)
        self.W_v = nn.Linear(d_pe, d_pe, bias=False)
        self.proj = nn.Linear(d_pe, dim)

    def forward(self, tokens, ctx: PEContext):
        image = ctx.extra["image"]
        ll, hs = self.fe(image)
        gates = torch.sigmoid(self.W_g(self.scale_emb)) * torch.tanh(self.W_v(self.scale_emb))  # (K, d_pe)
        acc = pool_to_grid(ll, ctx.grid) * gates[0].view(1, -1, 1, 1)
        k = 1
        for j in range(self.fe.levels):
            for o in range(3):
                acc = acc + pool_to_grid(hs[j][:, :, o], ctx.grid) * gates[k].view(1, -1, 1, 1)
                k += 1
        pe = self.proj(acc.flatten(2).transpose(1, 2))
        return tokens + pe.to(tokens.dtype)


# --------------------------------------------------------------------------- #
# Rotary machinery
# --------------------------------------------------------------------------- #
def dyadic_bands(n_freq: int, n_bands: int, finest_period: float = 2.0) -> torch.Tensor:
    """n_freq angular frequencies (rad / patch) grouped into n_bands dyadic
    octaves: band b spans periods [finest*2^b, finest*2^(b+1)). Returned finest
    -> coarsest, length n_freq, with band id = i // (n_freq // n_bands)."""
    per = n_freq // n_bands
    assert per * n_bands == n_freq, "n_freq must be divisible by n_bands"
    freqs = []
    for b in range(n_bands):
        lo, hi = finest_period * 2 ** b, finest_period * 2 ** (b + 1)
        periods = torch.logspace(math.log10(lo), math.log10(hi), per + 1)[:-1]
        freqs.append(2 * math.pi / periods)
    return torch.cat(freqs)


def theta_bands(n_freq: int, theta: float = 100.0) -> torch.Tensor:
    """RoPE-ViT default: f_i = theta^(-i / n_freq)."""
    return theta ** (-torch.arange(n_freq, dtype=torch.float32) / n_freq)


class Rotary2D(nn.Module):
    """2D rotary tables for one head_dim. `mode='axial'` (RoPE-ViT axial):
    first half of the pairs rotate with x, second half with y. `mode='mixed'`
    (RoPE-ViT mixed): every pair has learnable (f_x, f_y) per head, initialised
    axially. `init` selects the frequency grid: 'theta' (RoPE-ViT) or 'dyadic'
    (octave bands, used by ViPER so that gates act on wavelet-like scales)."""

    def __init__(self, head_dim: int, num_heads: int, mode: str = "axial", init: str = "theta",
                 theta: float = 100.0, n_bands: int = 4, depth: int = 1):
        super().__init__()
        assert head_dim % 4 == 0
        self.head_dim, self.num_heads, self.mode, self.depth = head_dim, num_heads, mode, depth
        n_pairs = head_dim // 2
        n_axis = n_pairs // 2
        base = theta_bands(n_axis, theta) if init == "theta" else dyadic_bands(n_axis, n_bands)
        # band b = contiguous block of n_axis // n_bands pairs on each axis (finest -> coarsest)
        assert n_axis % n_bands == 0, "pairs per axis must be divisible by n_bands"
        self.n_bands = n_bands
        fx = torch.cat([base, torch.zeros(n_axis)])
        fy = torch.cat([torch.zeros(n_axis), base])
        if mode == "mixed":
            # RoPE-ViT mixed: learnable (f_x, f_y) per pair, per head *and per layer*
            f = torch.stack([fx, fy], 0)[None, None].repeat(depth, num_heads, 1, 1)             # (L, H, 2, P)
            self.freqs = nn.Parameter(f)
        else:
            self.register_buffer("freqs", torch.stack([fx, fy], 0)[None, None], persistent=False)  # (1, 1, 2, P)

    def tables(self, grid: Tuple[int, int], device, scale: float = 1.0):
        """-> cos, sin of shape (L|1, H|1, N, D) in the rotate-half layout."""
        coords = grid_coords(grid, device, scale=scale)                       # (N, 2) [y, x]
        f = self.freqs.to(device)                                           # (L|1, H|1, 2, P)
        ang = coords[:, 1].view(1, 1, -1, 1) * f[:, :, 0, None, :] + coords[:, 0].view(1, 1, -1, 1) * f[:, :, 1, None, :]
        ang = torch.cat([ang, ang], dim=-1)                                  # (L|1, H|1, N, D)
        return ang.cos(), ang.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, N, D); cos/sin: (H|1, N, D) broadcast over batch."""
    return x * cos[None] + rotate_half(x) * sin[None]


class RoPEAxial(nn.Module):
    kind = "attention"

    def __init__(self, head_dim: int, num_heads: int, depth: int = 1, theta: float = 100.0, init: str = "theta",
                 n_bands: int = 4, lepe: bool = False):
        super().__init__()
        self.rot = Rotary2D(head_dim, num_heads, "axial", init, theta, n_bands)
        self.lepe = lepe                                           # + CSWin LePE on V (control for ViPER+LePE)

    def forward(self, tokens, ctx: PEContext):
        ctx.rope_cos, ctx.rope_sin = self.rot.tables(ctx.grid, tokens.device, ctx.extra.get("pos_scale", 1.0))
        ctx.lepe = self.lepe
        return tokens


class RoPEMixed(RoPEAxial):
    def __init__(self, head_dim: int, num_heads: int, depth: int = 1, theta: float = 100.0, init: str = "theta",
                 n_bands: int = 4, lepe: bool = False):
        nn.Module.__init__(self)
        self.rot = Rotary2D(head_dim, num_heads, "mixed", init, theta, n_bands, depth=depth)
        self.lepe = lepe


# --------------------------------------------------------------------------- #
# Attention-bias PEs
# --------------------------------------------------------------------------- #
def alibi_slopes(H: int):
    """ALiBi geometric slopes (Press et al.) incl. the non-power-of-2 rule, as in CROMA's code:
    H = 6 -> (2^-2, 2^-4, 2^-6, 2^-8, 2^-1, 2^-3)."""
    def pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start ** i for i in range(n)]
    if math.log2(H).is_integer():
        return pow2(H)
    c = 2 ** math.floor(math.log2(H))
    return pow2(c) + alibi_slopes(2 * c)[0::2][:H - c]


class ALiBi2D(nn.Module):
    """2D ALiBi (CROMA): logits -= slope_h * ||p_i - p_j||_2 (patch units), identical in every
    layer; CLS row/column 0 (CROMA/LookHere convention)."""
    kind = "attention"

    def __init__(self, num_heads: int):
        super().__init__()
        slopes = torch.tensor(alibi_slopes(num_heads))
        self.register_buffer("slopes", slopes, persistent=False)
        self._cache = {}

    def forward(self, tokens, ctx: PEContext):
        key = (ctx.grid, tokens.device)
        if key not in self._cache:
            c = grid_coords(ctx.grid, tokens.device)
            d = torch.cdist(c, c)                                         # (N, N)
            n, p = d.shape[0], ctx.n_prefix
            bias = torch.zeros(1, self.slopes.numel(), n + p, n + p, device=tokens.device)
            bias[:, :, p:, p:] = -self.slopes.view(1, -1, 1, 1) * d[None, None]
            self._cache = {key: bias}
        ctx.attn_bias = self._cache[key].to(tokens.dtype)
        return tokens


class RelBias(nn.Module):
    """Swin-style learned relative position bias table, (2h-1)(2w-1) x heads,
    bicubic-resampled (log-spaced like Swin-v2 is unnecessary at these grids)
    when evaluated at a different grid."""
    kind = "attention"

    def __init__(self, num_heads: int, grid: Tuple[int, int]):
        super().__init__()
        self.grid, self.num_heads = grid, num_heads
        self.table = nn.Parameter(torch.zeros((2 * grid[0] - 1) * (2 * grid[1] - 1), num_heads))
        nn.init.trunc_normal_(self.table, std=0.02)

    def _table_for(self, grid, device):
        if grid == self.grid:
            return self.table
        t = self.table.T.reshape(1, self.num_heads, 2 * self.grid[0] - 1, 2 * self.grid[1] - 1)
        t = F.interpolate(t, size=(2 * grid[0] - 1, 2 * grid[1] - 1), mode="bicubic", align_corners=True)
        return t.reshape(self.num_heads, -1).T

    def forward(self, tokens, ctx: PEContext):
        h, w = ctx.grid
        c = grid_coords(ctx.grid, tokens.device).long()
        rel = c[:, None, :] - c[None, :, :]
        idx = (rel[..., 0] + h - 1) * (2 * w - 1) + (rel[..., 1] + w - 1)   # (N, N)
        table = self._table_for(ctx.grid, tokens.device)
        b = table[idx.reshape(-1)].reshape(h * w, h * w, -1).permute(2, 0, 1)   # (H, N, N)
        n, p = h * w, ctx.n_prefix
        bias = F.pad(b, (p, 0, p, 0))[None]
        ctx.attn_bias = bias.to(tokens.dtype)
        return tokens


class LePE(nn.Module):
    """CSWin LePE: positional information injected as a depthwise conv on V
    inside every attention block (the convs live in PEAttention; this module just
    switches the flag)."""
    kind = "attention"

    def forward(self, tokens, ctx: PEContext):
        ctx.lepe = True
        return tokens


# --------------------------------------------------------------------------- #
# Conv PEG (CPVT)
# --------------------------------------------------------------------------- #
class PEG(nn.Module):
    """CPVT Positional Encoding Generator: zero-padded depthwise 3x3 conv on the
    token grid with a residual, inserted after the first encoder block
    (CPVT default) — `positions` chooses the block indices."""
    kind = "conv"

    def __init__(self, dim: int, positions: Tuple[int, ...] = (0,), k: int = 3):
        super().__init__()
        self.positions = set(positions)
        self.convs = nn.ModuleDict({str(p): nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim) for p in positions})

    def forward(self, tokens, ctx: PEContext):
        return tokens

    def after_block(self, i: int, x: torch.Tensor, ctx: PEContext) -> torch.Tensor:
        if i not in self.positions:
            return x
        p = ctx.n_prefix
        pre, t = x[:, :p], x[:, p:]
        B, N, C = t.shape
        g = t.transpose(1, 2).reshape(B, C, *ctx.grid)
        g = self.convs[str(i)](g) + g
        return torch.cat([pre, g.flatten(2).transpose(1, 2)], dim=1)


def haar_dwt(x: torch.Tensor):
    """One level of the orthonormal 2D Haar DWT on (B, C, H, W); odd sizes are zero-padded.
    -> ll, (lh, hl, hh), each (B, C, ceil(H/2), ceil(W/2))."""
    H, W = x.shape[-2:]
    if H % 2 or W % 2:
        x = F.pad(x, (0, W % 2, 0, H % 2))
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return (a + b + c + d) / 2, ((a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2)


def haar_idwt(ll: torch.Tensor, hs, size: Tuple[int, int]) -> torch.Tensor:
    """Inverse of `haar_dwt`, cropped to `size` (exact reconstruction)."""
    h1, h2, h3 = hs
    a, b = (ll + h1 + h2 + h3) / 2, (ll - h1 + h2 - h3) / 2
    c, d = (ll + h1 - h2 - h3) / 2, (ll - h1 - h2 + h3) / 2
    B, C, h, w = ll.shape
    top = torch.stack((a, b), -1).reshape(B, C, h, 2 * w)                      # interleave columns
    bot = torch.stack((c, d), -1).reshape(B, C, h, 2 * w)
    out = torch.stack((top, bot), -2).reshape(B, C, 2 * h, 2 * w)              # interleave rows
    return out[..., :size[0], :size[1]]


class WaveLePE(nn.Module):
    """Wavelet-domain locally-enhanced PE on the values. V (on the token grid) is
    decomposed by a J-level Haar DWT; every detail level is filtered by a
    depthwise 3x3 conv (one kernel per channel and level, shared by the three
    orientations, plus a per-orientation channel gain) and the coarsest
    approximation by its own depthwise 3x3 conv, then the inverse DWT maps the
    result back to the grid. Level j sees a (3*2^j)-patch neighbourhood, so the
    term carries direction-aware relative position at J+1 dyadic scales (LePE is
    the J=0 special case). With `gates` each scale's reconstruction is weighted
    per token, head and level (the same wavelet gate head that steers the rotary
    bands)."""

    def __init__(self, dim: int, num_heads: int, levels: int = 2, kernel: int = 3, det: str = "gain"):
        super().__init__()
        self.dim, self.num_heads, self.levels = dim, num_heads, levels
        # det="conv": depthwise 3x3 per level (shared by orientations); det="gain": the Haar details are
        # themselves fixed directional difference filters, so a learned per-channel gain per orientation
        # and level already gives a direction-aware multi-scale relative operator, at almost no cost.
        self.det = nn.ModuleList(nn.Conv2d(dim, dim, kernel, 1, kernel // 2, groups=dim) for _ in range(levels)) \
            if det == "conv" else None
        self.gain = nn.Parameter(torch.ones(levels, 3, dim))
        self.app = nn.Conv2d(dim, dim, kernel, 1, kernel // 2, groups=dim)

    def forward(self, v: torch.Tensor, grid: Tuple[int, int], gates: Optional[torch.Tensor] = None):
        """v: (B, H, N, D) patch values; gates: (B, H, N, J+1) = [approx, level 1 (fine) .. level J] or None
        -> (B, H, N, D). Gates are applied to the wavelet coefficients (token gates averaged over each
        coefficient's 2^j x 2^j support), so a single inverse DWT is needed."""
        B, Hh, N, D = v.shape
        x = v.transpose(1, 2).reshape(B, N, Hh * D).transpose(1, 2).reshape(B, Hh * D, *grid)
        if gates is not None:
            gm = gates.to(x.dtype).permute(0, 1, 3, 2).reshape(B, Hh, -1, *grid)   # (B, H, J+1, h, w)
        sizes, dets = [], []
        for j in range(self.levels):
            sizes.append(x.shape[-2:])
            x, hs = haar_dwt(x)
            hs = torch.stack(hs, 1)                                             # (B, 3, C, h, w)
            f = self.det[j](hs.flatten(0, 1)).view_as(hs) if self.det is not None else hs
            f = f * self.gain[j].to(hs.dtype)[None, :, :, None, None]
            if gates is not None:
                gm = F.avg_pool2d(gm.flatten(1, 2), 2, ceil_mode=True).view(B, Hh, -1, *x.shape[-2:])
                gj = gm[:, :, j + 1, None].expand(B, Hh, D, *x.shape[-2:]).reshape(B, 1, Hh * D, *x.shape[-2:])
                f = f * gj
            dets.append(f.unbind(1))
        c = self.app(x)
        if gates is not None:
            c = c * gm[:, :, 0, None].expand(B, Hh, D, *x.shape[-2:]).reshape(B, Hh * D, *x.shape[-2:])
        for j in range(self.levels - 1, -1, -1):
            c = haar_idwt(c, dets[j], sizes[j])
        return c.reshape(B, Hh, D, N).transpose(-1, -2).to(v.dtype)


class FeatureWaveGate(nn.Module):
    """Per-layer band gates from the wavelet energy of the layer's own tokens
    (instead of the pixels): x -> 1x1 (C->r) -> J-level Haar DWT -> log energies
    upsampled to the grid -> GroupNorm -> 1x1 + GELU -> 1x1 (zero init) -> 2*sigmoid."""

    def __init__(self, dim: int, num_heads: int, n_bands: int, levels: int = 2, r: int = 8, hidden: int = 32,
                 gate_max: float = 2.0):
        super().__init__()
        self.levels, self.num_heads, self.n_bands, self.gate_max = levels, num_heads, n_bands, gate_max
        self.reduce = nn.Conv2d(dim, r, 1)
        n_feat = r * (3 * levels + 1)
        self.norm = nn.GroupNorm(1, n_feat)
        self.hidden = nn.Sequential(nn.Conv2d(n_feat, hidden, 1), nn.GELU())
        self.head = nn.Conv2d(hidden, num_heads * n_bands, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        """x: (B, N, C) patch tokens -> (B, H, N, nb)."""
        B, N, C = x.shape
        y = self.reduce(x.transpose(1, 2).reshape(B, C, *grid).float())
        feats = []
        for _ in range(self.levels):
            y, hs = haar_dwt(y)
            for h in hs:
                feats.append(F.interpolate(h ** 2, size=grid, mode="bilinear", align_corners=False))
        feats.append(F.interpolate(y ** 2, size=grid, mode="bilinear", align_corners=False))
        f = self.norm(torch.log(torch.cat(feats, 1) + 1e-6))
        g = self.head(self.hidden(f))                                           # (B, H*nb, h, w)
        g = g.reshape(B, self.num_heads, self.n_bands, -1).transpose(-1, -2)    # (B, H, N, nb)
        return self.gate_max * torch.sigmoid(g)


# --------------------------------------------------------------------------- #
# ViPER (ours): wavelet-steered multi-scale rotary PE
# --------------------------------------------------------------------------- #
class ViPER(nn.Module):
    """Wavelet-steered rotary positional encoding.

    Relative position enters attention through 2D rotary phases (RoPE-ViT theta
    grid, mixed learnable frequencies by default) whose frequency pairs are
    grouped into B contiguous bands, i.e. a multi-scale decomposition of the
    relative-position kernel. (A dyadic grid, periods 2..32 patches, collapses
    beyond the training resolution; kept as the `rope_init="dyadic"` ablation.) A tiny
    gating head reads the per-token local wavelet energy (per DWT level and
    orientation, pooled to the patch grid) and outputs one gate per head and
    band that rescales the *query's* rotated components of that band:

        logit_ij = sum_b g_b(i) * < R_b(p_i) q_i^b , R_b(p_j) k_j^b >
                 = sum_b g_b(i) * K_b(p_i - p_j ; q_i, k_j)

    so each query chooses the positional scales at which it measures distance,
    while translation equivariance of every band is preserved exactly (the gate
    depends on content, not on absolute position) and no table is needed, so
    any test resolution is supported. By default (gate_res=0) the wavelet branch reads the image at its
    native resolution, so every token's wavelet statistics cover the same 16x16-pixel footprint at any test
    size; gate_res>0 resamples the image to that size first (the setting of runs before 2026-11-02).
    """
    kind = "attention"

    def __init__(self, head_dim: int, num_heads: int, depth: int, levels: int = 4, wavelet: str = "db4",
                 n_bands: int = 4, rope_mode: str = "mixed", rope_init: str = "theta", theta: float = 100.0,
                 gate_res: int = 0, gate_hidden: int = 32, gate_k: bool = False, per_layer: bool = True,
                 gate_max: float = 2.0, use_ll: bool = True, ctx_conv: bool = True, fast_dwt: bool = True,
                 use_gate: bool = True, lepe: bool = False, dim: Optional[int] = None,
                 gate_src: str = "pixel", wave_v: bool = False, wave_v_levels: int = 2, wave_v_gate: bool = False,
                 wave_v_det: str = "gain", feat_levels: int = 2, string_p: bool = False,
                 score_drop_fine: int = 0, grid: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.score_drop_fine = score_drop_fine                     # zero the k finest score-path bands (range split)
        if string_p:                                               # Cayley-STRING basis: R(p) P z, P orthogonal
            self.S_raw = nn.Parameter(torch.zeros(depth, num_heads, head_dim, head_dim))   # S = 0 -> P = I
        self.lepe = lepe                                           # + CSWin LePE (depthwise 3x3 conv on V)
        self.gate_src, self.wave_v_gate = gate_src, wave_v_gate
        if wave_v:                                                 # wavelet-domain LePE on the values
            self.wv = nn.ModuleList(WaveLePE(dim, num_heads, wave_v_levels, det=wave_v_det) for _ in range(depth))
        if gate_src == "feature":                                  # band gates from the wavelet energy of tokens
            self.fg = nn.ModuleList(FeatureWaveGate(dim, num_heads, n_bands, feat_levels) for _ in range(depth))
            use_gate = wave_v_gate                                 # pixel branch only if it must gate WaveLePE
        self.rot = Rotary2D(head_dim, num_heads, rope_mode, rope_init, theta, n_bands,
                            depth=depth if rope_mode == "mixed" else 1)
        self.levels, self.n_bands, self.num_heads, self.depth = levels, n_bands, num_heads, depth
        self.gate_res, self.gate_k, self.per_layer, self.gate_max = gate_res, gate_k, per_layer, gate_max
        self.use_ll, self.use_gate = use_ll, use_gate
        if not use_gate:                                          # no wavelet branch at all (and no params)
            return
        if gate_src == "static":                                  # control: learned per-(layer, head, band) gates,
            L = depth if per_layer else 1                         # no dependence on the image
            self.static_z = nn.Parameter(torch.zeros(L, num_heads, n_bands))   # zero-init -> gate = 1
            self.n_rope_g, self.n_v_g = n_bands, 0
            return
        n_feat = 3 * levels + (1 if use_ll else 0)
        if gate_src == "rgb":                                     # control: learned patch filters on RGB instead
            self.rgb = nn.Conv2d(3, n_feat, 16, 16)                # of the fixed wavelet energies (same width)
        else:
            self.fe = FastDWT(levels, wavelet) if fast_dwt else WaveletFrontEnd(levels, wavelet)
        self.norm = nn.GroupNorm(1, n_feat, affine=True)          # per-image standardisation of log-energies
        self.ctx_conv = nn.Conv2d(n_feat, n_feat, 3, 1, 1, groups=n_feat) if ctx_conv else nn.Identity()
        self.hidden = nn.Sequential(nn.Conv2d(n_feat, gate_hidden, 1), nn.GELU())
        self.n_rope_g = 0 if gate_src == "feature" else n_bands
        self.n_v_g = wave_v_levels + 1 if (wave_v and wave_v_gate) else 0
        n_out = num_heads * (self.n_rope_g + self.n_v_g) * (depth if per_layer else 1)
        self.head = nn.Conv2d(gate_hidden, n_out, 1)
        nn.init.zeros_(self.head.weight)                          # start as plain multi-band RoPE (gate = 1)
        nn.init.zeros_(self.head.bias)

    def energy_features(self, image: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        if self.gate_src == "rgb":
            return self.norm(self.rgb(image)[..., :grid[0], :grid[1]])
        if self.gate_res and (image.shape[-1] != self.gate_res or image.shape[-2] != self.gate_res):  # 0: native res
            image = F.interpolate(image.float(), size=(self.gate_res, self.gate_res), mode="bilinear",
                                  align_corners=False, antialias=True)
        ll, hs = self.fe(image)
        feats = []
        for j in range(self.levels):
            e = hs[j][:, 0] ** 2                                   # (B, 3, h_j, w_j)
            feats.append(pool_to_grid(e, grid))
        if self.use_ll:
            feats.append(pool_to_grid(ll ** 2, grid))
        f = torch.log(torch.cat(feats, 1) + 1e-6)
        return self.norm(f)

    def gates(self, image: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        """-> (B, L|1, H, N, n_bands) in [0, gate_max]."""
        if self.gate_src == "static":
            z = self.static_z[None, :, :, None, :]                 # (1, L, H, 1, nb)
            return (self.gate_max * torch.sigmoid(z)).expand(image.shape[0], -1, -1, grid[0] * grid[1], -1)
        f = self.energy_features(image, grid)
        f = self.ctx_conv(f)
        g = self.head(self.hidden(f))                              # (B, L*H*nb, h, w)
        B = g.shape[0]
        L = self.depth if self.per_layer else 1
        g = g.reshape(B, L, self.num_heads, self.n_rope_g + self.n_v_g, -1).permute(0, 1, 2, 4, 3)
        return self.gate_max * torch.sigmoid(g)

    def forward(self, tokens, ctx: PEContext):
        ctx.rope_cos, ctx.rope_sin = self.rot.tables(ctx.grid, tokens.device, ctx.extra.get("pos_scale", 1.0))
        ctx.lepe = self.lepe
        ctx.rope_gate_k = self.gate_k
        if hasattr(self, "S_raw"):
            eye = torch.eye(self.S_raw.shape[-1], device=tokens.device)

            def pre(qp, kp, layer):
                with torch.autocast("cuda", enabled=False):
                    Sr = self.S_raw[layer].float()
                    P = torch.linalg.solve(eye + Sr - Sr.transpose(-1, -2), eye - Sr + Sr.transpose(-1, -2))
                P = P.to(qp.dtype)
                return torch.einsum("hij,bhnj->bhni", P, qp), torch.einsum("hij,bhnj->bhni", P, kp)
            ctx.pre_rot_fn = pre
        if self.gate_src == "feature":
            grid = ctx.grid
            ctx.gate_fn = lambda x, layer: self.fg[layer](x, grid)
        vg = None
        if self.use_gate:
            g = self.gates(ctx.extra["image"], ctx.grid)           # (B, L|1, H, N, nb [+ J+1])
            if self.n_rope_g:
                # bands are contiguous blocks of frequency pairs on each axis, so the attention layer
                # applies the gate with a reshape (B,H,N,2,2,nb,per) — no gather over the head dim.
                ctx.rope_gate = g[..., :self.n_rope_g].to(tokens.dtype)
            if self.n_v_g:
                vg = g[..., self.n_rope_g:]
            ctx.extra["gates_raw"] = g
        if hasattr(self, "wv"):
            grid = ctx.grid
            ctx.value_fn = lambda v, layer: self.wv[layer](
                v, grid, None if vg is None else vg[:, layer if vg.shape[1] > 1 else 0])
        if self.score_drop_fine and ctx.rope_gate is not None:
            g = ctx.rope_gate.clone()
            g[..., :self.score_drop_fine] = 0                      # short range left to the value path
            ctx.rope_gate = g
        return tokens


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def build_pe(name: str, *, dim: int, head_dim: int, num_heads: int, depth: int, grid: Tuple[int, int],
             **kw) -> nn.Module:
    name = name.lower()
    if name == "none":
        return NoPE()
    if name in ("learned", "ape"):
        return LearnedAPE(dim, grid)
    if name == "sincos":
        return SinCos2D(dim)
    if name == "fourier":
        return LearnableFourierPE(dim, **kw)
    if name == "wave_add":
        return WaveAdditivePE(dim, **kw)
    if name == "rope_axial":
        return RoPEAxial(head_dim, num_heads, depth, **kw)
    if name == "rope_mixed":
        return RoPEMixed(head_dim, num_heads, depth, **kw)
    if name == "alibi":
        return ALiBi2D(num_heads)
    if name == "relbias":
        return RelBias(num_heads, grid)
    if name == "lepe":
        return LePE()
    if name in ("cpe", "peg"):
        return PEG(dim, **kw)
    if name == "viper":
        return ViPER(head_dim, num_heads, depth, dim=dim, grid=grid, **kw)
    from .pe_extra import build_extra
    m = build_extra(name, dim=dim, head_dim=head_dim, num_heads=num_heads, depth=depth, grid=grid, **kw)
    if m is None:
        raise ValueError(f"unknown PE '{name}'")
    return m


PE_NAMES = ["none", "learned", "sincos", "fourier", "wave_add", "rope_axial", "rope_mixed",
            "alibi", "relbias", "lepe", "cpe", "viper",
            "liere", "comrope_ld", "comrope_ap", "string_cayley", "spiral", "lookhere", "pape",
            "irpe", "dpb", "gpsa", "wavelet_rpe", "carope"]
