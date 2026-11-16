# Positional-encoding baselines: faithful re-implementation spec

Extracted from the official repositories (raw.githubusercontent.com) and the arXiv papers.
Every claim is tagged **[code]** (read from the official source file), **[paper]** (read from the
arXiv HTML/PDF text), **[ckpt]** (read from a released checkpoint), or **[inferred]/[UNVERIFIED]**.

Common conventions used in all sketches below:

* `q, k : (B, H, N, d)` with `d = head_dim` (64 for DeiT-S/B/ViT-B), token 0 = CLS.
* `coords : (N, 2)` per-token patch-grid coordinates `(y, x)` (row, col); the CLS row is special-cased
  per method. Each method says which units it expects (integer index, `[0,1]`, centred, ...).
* "q/k transform" hooks return `(q', k')`; "bias" hooks return a tensor broadcastable to `(B, H, N, N)`
  that is **added to the logits** (sign already applied).
* Scores in all original codebases are `softmax(q'k'^T * d^-0.5 (+bias))`; PaPE is the one method whose
  bias sits *inside* the `d^-0.5` scaling (see §6).

Verification summary

| Method | Formula | Init | Defaults (heads/layers/size) | Coordinates / CLS |
|---|---|---|---|---|
| LieRE | [code] | [code] | [code]+[paper] (block size: paper says 8, repo default 64) | [code] |
| ComRoPE-AP / LD | [code] | [code] (init_std default; paper silent) | [code]+[paper] (b=8 paper; code default 2) | [code] |
| STRING Cayley / Circulant | [paper] | **UNVERIFIED** (not in paper) | [paper] partially; per-layer sharing, coords, CLS **UNVERIFIED** | **UNVERIFIED** |
| Spiral RoPE | [code]+[paper] | n/a (fixed) | [code]+[paper] (K: paper 16, repo script default 4) | [code] |
| LookHere | [code]+[paper]+[ckpt] | n/a (fixed) | [ckpt] for 12 heads; 6-head path analysed from code (partly broken) | [code] |
| PaPE / PaPE-RI | [code]+[paper] | [code] | [code]+[paper] | [code] |
| 2D-ALiBi | [code] (CROMA, PaPE repo) | n/a | [code] | [code]; LookHere's own ALiBi variant partly [inferred] |
| iRPE (ctx/product/K) | [code] | [code] (zero tables) | [code] (released DeiT-S config) | [code] (skip bucket); APE kept [code] |
| CrossFormer DPB | [code] | [code] | [code] | [code]; CLS = 0 **[assumption]** (CrossFormer has no CLS) |
| ConViT GPSA | [code] | [code] | [code] (6-head layout follows code, not a released model) | [code]; CLS handling **deviates** (CLS present from block 0) |
| Wavelet RPE (2D port) | [paper] (no code released) | n/a (fixed) | [paper] 1D; 2D port + scale set **[ours]** | CLS = 0 **[ours]** |

---

## 1. LieRE (ICML 2025, arXiv 2406.10322, github.com/StanfordMIMI/LieRE)

Sources: `models/rope_vit.py` (`LierePositionEncoder`, `FlexibleAttention`, `PositionEncoderBase`),
`models/model_args.py`, `lightning_modules/lightning_liere_image_classification.py`,
`lightning_conf/{imagenet_224,cifar100}.yaml`, paper §4, §5.4, App. B/C.

**Coordinates [code].** For an `n_y x n_x` patch grid, `PositionEncoderBase.forward` builds
`cartesian_prod(linspace(0, n_y/n_y_train, n_y), linspace(0, n_x/n_x_train, n_x))`, i.e. at training
resolution `p = (y/(n_y-1), x/(n_x-1)) ∈ [0,1]^2` (14x14 grid -> spacing 1/13). At a different test
resolution the range grows proportionally (extrapolation, not interpolation). Order: first axis = rows.

**Parameters [code].** `generator_raw_params ∈ R^{A × L' × (H'·G) × g × g}` with
`A = 2` (input dimensionality), `L' = depth` if `rotary_embedding_per_layer` else 1,
`H' = heads` if `rotary_embedding_per_head` else 1, `g = generator_dim` (block size),
`G = head_dim // g` (blocks per head). Init: `torch.rand(...) * 2π` (elementwise U(0, 2π), "inspired
by RoPE-Mixed"). Effective skew-symmetric basis (the `-0.5` in the code cancels exactly):

    U_a = triu(raw_a, diagonal=1);   A_a = U_a - U_a^T        (a = y, x)      [g×g skew-symmetric]

so only the strict upper triangle (U(0,2π)) matters: `g(g-1)/2` effective params per block/axis.

**Rotation [code].** For token i with coordinates `p_i`:

    M_i = Σ_a p_{i,a} A_a           (per layer l, head h, block j)         [g×g]
    R_i = blockdiag_j( matrix_exp(M_i^{(l,h,j)}) )                        [d×d, in SO(d)]
    q'_i = R_i q_i,   k'_i = R_i k_i                                       (both q and k, same R)
    score_ij = q_i^T R_i^T R_j k_j · d^{-1/2}

`matrix_exp` is evaluated in fp32 (`EXPONENTIATE_ONE_SHOT`) then cast back. CLS: its generator is set to
zero (`cls_generator = zeros`) -> identity. No APE: `pos_embedding` is zeroed and frozen and
`enable_ape=False` for the `liere` model. Note that for `g > 2` the A_y, A_x of a block do **not**
commute in general, so `R_i^T R_j ≠ R(p_j − p_i)` exactly; the paper accepts this (STRING §2 points
it out).

**Defaults.**
* Paper [paper]: "By default, the skew symmetric bases are learned separately for every layer and
  attention head"; block-size ablation {2,4,8,16,32,48,64} peaks "around 8x8"; the headline variant is
  **LieRE8** (g = 8); dense LieRE64 reported slightly worse; "≈580k extra parameters (0.68% of ViT-B)"
  which corresponds to g = 64 (2·12·12·2016 = 580,608).
* Repo [code]: `ModelArgs.generator_dim = 64` (dense) is the code default; imagenet/cifar configs set
  `rotary_embedding_per_layer: True`, `rotary_embedding_per_head: True`; `dim_head` asserted = 64;
  phase_type "naver" (= learned); RoPE-Mixed baseline = same class with `generator_dim=2`.
* Param counts (effective, A=2): `2·L·H·G·g(g−1)/2`. ViT-B g=8: 64,512; DeiT-S (H=6) g=8: 32,256;
  g=64: 580,608 / 290,304. Stored tensors are g² per block (lower triangle unused).

**Recommended for your wrapper:** g = 8, per layer, per head, coords in [0,1] as above, CLS identity,
no APE. Also run g = 64 if you want the "LieRE64" row.

```python
class LieRE(nn.Module):                      # one instance per layer
    def __init__(self, heads, head_dim=64, g=8, n_axes=2, per_head=True):
        super().__init__()
        self.g, Hp = g, (heads if per_head else 1)
        self.raw = nn.Parameter(torch.rand(n_axes, Hp, head_dim // g, g, g) * 2 * math.pi)
    def forward(self, q, k, coords, cls=True):          # coords (N, 2) in [0,1]
        U = torch.triu(self.raw, 1); A = U - U.transpose(-1, -2)          # skew-symmetric bases
        c = coords.float().clone()
        if cls: c[0] = 0                                                    # CLS -> identity
        M = torch.einsum('na,ahgij->hngij', c, A).contiguous()             # (Hp,N,G,g,g)
        R = torch.linalg.matrix_exp(M).to(q.dtype)                         # fp32 exp, block-diag
        def apply(x):                                                      # x (B,H,N,d)
            xs = x.reshape(*x.shape[:3], -1, self.g, 1)                    # (B,H,N,G,g,1)
            return (R @ xs).reshape(x.shape)
        return apply(q), apply(k)
```

---

## 2. ComRoPE (CVPR 2025, arXiv 2506.03737, github.com/Longin-Yu/ComRoPE)

Sources: `model.py` (`CustomVITConfig.get_positions`, `RoPESelfAttentionBase`, `ComRoPEAPAttention`,
`ComRoPELDAttention`, `CustomCLIPVisionEmbeddings`), README, paper §3.3, §4.1.1, App. E.

**Coordinates [code].** `meshgrid(arange(n), arange(n), indexing='ij')` -> `(row, col)`; with
`axis_scale=True` (default) `p = (idx + 0.5) / n ∈ (0,1)` (patch centres, "divide by image size").
CLS is **prepended with position (0.5, 0.5)** (`prepend_cls=True, cls_position=0.5`), i.e. the CLS
token is rotated like a patch at the image centre (not identity). Optional training augmentation:
`positions += clamp(N(0,1)·σ·0.5, −0.5, 0.5)` in patch units before scaling (σ = "perturbation
intensity"; paper evaluates σ ∈ {0, 1}; σ=1 helps ComRoPE-LD slightly). Also an optional random global
offset (`offset_intensity`, default 0).

**Rotation [code]** (`get_rotation_matrix`): generators `A ∈ R^{H × A × G × b × b}` (G = d/b blocks),
`lnR_i = Σ_a p_{i,a} A_a` per block, `R_i = blockdiag(matrix_exp(lnR_i))` (fp32), `q' = R q`,
`k' = R k` (both). Because every block's axis generators commute by construction,
`R_i^T R_j = R(p_j − p_i)` holds exactly.

**ComRoPE-AP [code+paper]** (axial partition): one trainable `P ∈ R^{H × 1 × G × b × b}`,
`S = P − P^T`; block j is assigned to axis `(j mod A)`; `A_a[j] = S[j]` if `j ≡ a (mod A)` else 0.
Requires `d % A == 0` and `(d/A) % b == 0`. Effective params/head/layer: `G·b(b−1)/2`
(ViT-B, b=8: 8·28 = 224 per head, 2,688 per layer; DeiT-S: 1,344 per layer). Paper Table 7: `L·d·b`.

**ComRoPE-LD [code+paper]** (linearly dependent): `P ∈ R^{H × 1 × G × b × b}` plus
`multiplier θ ∈ R^{1 × A × G × 1 × 1}` (**shared across heads, one scalar per (axis, block)** in code;
paper writes θ_i per axis); `A_a[j] = θ_{a,j} · (P[j] − P[j]^T)`. Paper Table 7: `L·d·(b + A/b)`.

**Init [code].** `P ~ N(0, init_std)` with `init_std = 1.0` default (`CustomVITConfig`); LD multiplier
`~ N(0, 1)`. Paper: trained from scratch with random init (no std given); for fine-tuning a pre-trained
model "we initialize the angle matrix to zero" (identity rotation). `reset_pe()` re-draws these.

**Defaults.** Paper §4.1.1: "A block size of 8 is used in practice" (ablation b ∈ {2,4,8}); ViT-B/16,
224px, 12 heads, from scratch 200 epochs. Code: `block_size=2` is only the dataclass default; every
layer gets its own attention module (per-layer, per-head parameters). `theta_base` is for the vanilla
RoPE baseline only. **APE caveat [code]:** the flag is inverted (`keep_ape=True` selects
`CustomCLIPVisionEmbeddingsWithoutPE`); with the default `keep_ape=False` the model *does* add a
learnable grid-sampled APE. Paper §4.1.1 says "The APE codebook is removed for methods that do not
employ APE", so for a faithful ComRoPE run use **no APE**. The `q_proj` output is pre-multiplied by
`scale` before rotation (irrelevant, rotation is linear).

```python
class ComRoPE(nn.Module):                    # one instance per layer; variant 'AP' or 'LD'
    def __init__(self, heads, head_dim=64, b=8, n_axes=2, variant='AP', init_std=1.0):
        super().__init__()
        self.b, self.variant, G = b, variant, head_dim // b
        self.P = nn.Parameter(torch.randn(heads, 1, G, b, b) * init_std)
        if variant == 'LD':
            self.theta = nn.Parameter(torch.randn(1, n_axes, G, 1, 1))        # shared over heads
        else:
            assert G % n_axes == 0
            m = torch.zeros(n_axes, G)
            for a in range(n_axes): m[a, a::n_axes] = 1                        # block j -> axis j mod A
            self.register_buffer('mask', m.view(1, n_axes, G, 1, 1))
    def forward(self, q, k, coords):                 # coords (N,2) = ((row+.5)/n,(col+.5)/n); CLS=(.5,.5)
        S = self.P - self.P.transpose(-1, -2)                                  # skew-symmetric
        A = S * (self.theta if self.variant == 'LD' else self.mask)            # (H,A,G,b,b), commuting
        M = torch.einsum('na,hagij->hngij', coords.float(), A).contiguous()    # (H,N,G,b,b)
        R = torch.linalg.matrix_exp(M).to(q.dtype)
        def apply(x):
            xs = x.reshape(*x.shape[:3], -1, self.b, 1)
            return (R @ xs).reshape(x.shape)
        return apply(q), apply(k)
```

---

## 3. STRING: Cayley-STRING and Circulant-STRING (ICML 2025, arXiv 2502.02562)

No official code was found (web search; the only public implementation is a third-party
`CayleySTRING` module in HF `RationAI/LSP-DETR/modeling.py`, which I used only as a sanity check of
the operator order). Everything below is from the paper text (HTML + PDF, Def. 3.1, Thm 3.4, Eq. 9–11,
Thm 3.5, App. A.4, App. E).

**General form [paper, Def. 3.1 / Eq. 6–8].** `R(r_i) = exp(Σ_{k=1}^{d_c} L_k [r_i]_k)` with learnable,
pairwise-commuting skew-symmetric generators `L_k`; applied as `z_i -> R(r_i) z_i` to queries and
keys. Thm 3.4: any such R equals `P · RoPE(r) · P^T` for some orthogonal P; in a transformer the first P
cancels in `q^T k`, so "it is sufficient to take `R(r_i) = RoPE(r_i) P`, with P ∈ O(d) learnable"
(footnote 3: they redefine P^T as the learnable matrix). P is "shared between queries and keys" and
"learned and stored once per attention head".

**Cayley-STRING [paper, Eq. 10].** `P_Cayley = (I_d − S)(I_d + S)^{-1}`, S a "learnable (potentially
sparse) antisymmetric matrix"; `(I+S)^{-1} z` may be done with a linear solve. "For Cayley-STRING, the
angle-defining frequencies for RoPE *and* S are learned." Pipeline: `z -> P z -> RoPE(r)(P z)`
(basis change first, then the sparse 2x2 rotations). RoPE part for ViT [paper App. E]: RoPE-Mixed-style
(learned per-pair 2-D frequencies), "we initialize with 100 max wavelength as suggested in Heo et al.,
which we also adopt for STRING"; vanilla RoPE baseline splits dims between the 2 axes.
Sharing: "the default of learning them all separately" per head; sharing across heads "could yield
slight gains". Model: ViT-B/16, 224px, 300 epochs, batch 4096, lr 1e-3, Adam.

**Circulant-STRING [paper, Eq. 11, Thm 3.5, App. A.4].** `L_k = C_k − C_k^T`, `C_k` a learnable
circulant matrix given by d scalars `(c_0..c_{d-1})` (Eq. 11: first column `(c_0, c_1, ..., c_{d-1})`,
first row `(c_0, c_{d-1}, ..., c_1)`). Circulant matrices commute, so translation invariance is exact.
Computation (App. A.4): `C = DFT diag(DFT c) DFT^{-1}`, hence
`exp(Σ_k r_k L_k) z = DFT diag(exp(DFT u)) DFT^{-1} z` with `u = Σ_k r_k (c^{(k)} − t^{(k)})`,
`t` = first column of `C^T` (= first row of C), exp elementwise; i.e. `R z = ifft( exp(fft(u)) ⊙ fft(z) )`.
Since the circulant generator is skew-symmetric, `fft(u)` is purely imaginary, so this is exactly RoPE in
the DFT basis with per-axis frequencies `ω_m^{(k)} = Im fft(l^{(k)})_m` (`ω_0 = ω_{d/2} = 0`,
`ω_{d−m} = −ω_m`): d scalars per axis but only d/2−1 free frequencies. App. E: "for Circulant-STRING we
swept over block sizes in {4, 8, 16, 32, 64} (often around 16)"; **16 best on ImageNet/Places365,
32 on WebLI-3D**. [inferred] "block size" = size b of circulant blocks applied block-diagonally over the
head dim (b = 64 = whole head); the paper never defines it explicitly.

**UNVERIFIED / not in the paper:** init of S (I recommend S = 0 → P = I so training starts exactly at
RoPE-Mixed; LSP-DETR instead uses `torch.nn.utils.parametrizations.orthogonal(..., 'cayley')`, i.e.
random orthogonal init); whether S is dense (I use dense 64x64 per head: 2016 params/head/layer);
init of the circulant scalars (nothing stated; a sensible choice is to init `c` so that `Im fft(l)`
reproduces the RoPE-Mixed frequency schedule, or small random); per-layer vs shared-across-layers
(assume per layer, like RoPE-Mixed); coordinate units (assume integer patch indices as in RoPE-Mixed);
CLS handling (assume angle 0 → identity, as in RoPE-Mixed/rope-vit); whether an APE is kept (assume no).

```python
class CayleySTRING(nn.Module):           # R(r) z = RoPE_mixed(r) (P z);  P = (I-S)(I+S)^-1, per head
    def __init__(self, heads, head_dim=64, base=100.0):
        super().__init__()
        self.d = head_dim
        self.S_raw = nn.Parameter(torch.zeros(heads, head_dim, head_dim))      # S=0 -> P=I  [ASSUMED]
        mag = 1.0 / base ** (torch.arange(0, head_dim, 4).float() / head_dim)  # RoPE-Mixed init (rope-vit)
        ang = torch.rand(heads) * 2 * math.pi                                   # random per-head axis angle
        fx = torch.cat([mag[None] * ang.cos()[:, None], mag[None] * (ang + math.pi / 2).cos()[:, None]], 1)
        fy = torch.cat([mag[None] * ang.sin()[:, None], mag[None] * (ang + math.pi / 2).sin()[:, None]], 1)
        self.freqs = nn.Parameter(torch.stack([fy, fx], 0))                    # (2,H,d/2), learned
    def P(self):
        S = self.S_raw - self.S_raw.transpose(-1, -2); I = torch.eye(self.d, device=S.device)
        return torch.linalg.solve(I + S, I - S)       # = (I+S)^-1 (I-S) = (I-S)(I+S)^-1, orthogonal
    def forward(self, q, k, coords):                  # coords (N,2) integer (y,x); CLS row zeroed
        c = coords.float().clone(); c[0] = 0
        ang = torch.einsum('na,ahf->hnf', c, self.freqs)                       # (H,N,d/2)
        cos, sin, P = ang.cos().to(q.dtype), ang.sin().to(q.dtype), self.P().to(q.dtype)
        def apply(x):                                                           # (B,H,N,d)
            x = torch.einsum('hij,bhnj->bhni', P, x)                           # basis change first
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)
        return apply(q), apply(k)

class CirculantSTRING(nn.Module):        # exp(sum_k r_k (C_k - C_k^T)) z per b-block, via FFT
    def __init__(self, heads, head_dim=64, b=16, n_axes=2, init_scale=0.02):
        super().__init__()
        self.b = b
        self.c = nn.Parameter(torch.randn(n_axes, heads, head_dim // b, b) * init_scale)  # [INIT UNVERIFIED]
    def forward(self, q, k, coords):
        c = self.c
        l = c - torch.roll(c.flip(-1), 1, dims=-1)         # first column of C - C^T: l_j = c_j - c_{(-j) mod b}
        lam = torch.fft.fft(l.float(), dim=-1).imag        # (A,H,G,b) eigenvalues are purely imaginary
        cc = coords.float().clone(); cc[0] = 0
        ph = torch.einsum('na,ahgm->hngm', cc, lam)                            # (H,N,G,b) rotation angles
        E = torch.polar(torch.ones_like(ph), ph)                                # exp(i*ph), |E|=1
        def apply(x):
            xs = x.float().reshape(*x.shape[:3], -1, self.b)                   # (B,H,N,G,b)
            y = torch.fft.ifft(E[None] * torch.fft.fft(xs, dim=-1), dim=-1).real
            return y.reshape(x.shape).to(x.dtype)
        return apply(q), apply(k)
```

---

## 4. Spiral RoPE (ICML 2026, arXiv 2602.03227, github.com/huajianduzhuo-code/Spiral_RoPE)

Sources: `RoPE_DeiT/rope.py` (`VisionRotaryEmbedding`, `rotate_freqs`, `rotate_half`),
`RoPE_DeiT/models_v2.py` (`Attention`, `vit_models`), `RoPE_DeiT/run_all.sh`, `RoPE_DeiT/README.md`,
paper §4 (Eq. 8–13), App. A.1.

**What it is [paper].** Fixed (non-learned) 2-D RoPE with K directions `φ_k = kπ/K`,
`u_k = (cos φ_k, sin φ_k)`, projected coordinate `t_k(p) = p·u_k` (Eq. 8–10). The d channels are split
into K groups; group k is rotated by `t_k(p)` with its own frequency subset (Eq. 11). Base pool
`θ_t = θ_base^{−t/(d/4)}, t = 0..d/4−1` (Eq. 12), scaled by 1.5 (§4.1). Frequencies are taken in
adjacent pairs and dealt round-robin to the perpendicular direction pairs `(k, k+K/2)`:
`Θ^{(k)} = Θ^{(k+K/2)} = {θ_{2k}, θ_{2k+1}, θ_{2k+K}, θ_{2k+K+1}, ...}` (Eq. 13). K = 2 is axial RoPE.

**Exact code behaviour [code]** (`VisionRotaryEmbedding(dim=head_dim//2, pt_seq_len=img/patch,
rotate=R, max_freq_lang=s, min_freq_lang=0, same_theta=False, theta=10000)`):
* `n_f = d/4` frequencies `ω_t = s · 10000^{−t/(d/4)}` (geometric from `s` down to
  `s·10000^{−(n_f−1)/n_f}`), `s = 1.5` (run_all.sh default `MAX_FREQ_LANG=1.5`; the `-h` help text
  says 0.5, but the executed default is 1.5; paper: "frequency scale 1.5"). θ_base = 10000 in code (the
  paper's App. A.1 "15000" is not what the code does).
* Positions: integer patch indices `t = arange(n) · pt_seq_len / n` (0..13 at 224/16; rescaled to the
  training range at other resolutions = position interpolation).
* Channel layout per head (d = 64): channels `[0, d/2)` = "x-half" (first grid axis in the code is
  the row index, applied via `freqs[:, None, :]`), `[d/2, d)` = "y-half"; adjacent channel pairs
  `(2c, 2c+1)` share one frequency (`rotate_half` pairs adjacent channels). Within each half, pair t
  carries `ω_t`.
* Spiral: `R = rotate = K/2`. The 64-channel angle field is viewed as groups of `4R` channels; inside
  a group, sub-block i (4 channels = 2 frequency pairs) has its *spatial* angle map rotated by
  `i · 90°/R` using `F.grid_sample` (bilinear, `align_corners=True`, `padding_mode='border'`) about the
  grid centre `c = (n−1)/2`. For the linear field this equals `ω·(c + (p − c)·u)` **clamped so that
  `c + (p−c)·u ∈ [0, n−1]`** (border padding) — the official code therefore deviates from the ideal
  `p·u_k` near the image corners; the constant `c·ω` cancels in `q·k`. Resulting direction set is
  `{0, −90/R, ..., −(R−1)90/R} ∪ {90, 90−90/R, ...}` = K equally spaced directions over 180°, and the
  frequency/direction pairing reproduces Eq. 13 (verified by index arithmetic: x-half pair t → direction
  `(t // 2) mod R`, y-half pair t → direction `R + (t // 2) mod R`). Requires `d % (2K) == 0`.
* Applied to q and k of **patch tokens only**; CLS is left untouched (`cat((q[:, :1], rope(q[:, 1:])))`).
  Shared across heads (broadcast) and layers (identical buffers). The learnable APE (`pos_embed`,
  DeiT-III style) is **kept** (paper §4.1 "combined with the original absolute positional embedding").
* `same_theta=True` is the ablation where every direction reuses the same `d/(2K)` frequencies.

**Defaults.** Paper (classification, DeiT-S/B/L, 224px, 300 ep): **K = 16, scale 1.5**;
segmentation K = 16; DiT K = 8/6. Repo `run_all.sh` default is `ROTATE=2` (K = 4) with the DeiT-B
model — a discrepancy with the paper; use K = 16 (`ROTATE=8`) for the paper setting. For d = 64 and
K = 16 each direction gets exactly 2 frequencies `{ω_{2k}, ω_{2k+1}}` (4 channels).
DeiT-S: d = 64 too, so identical hyper-parameters.

```python
class SpiralRoPE(nn.Module):             # fixed; shared over heads/layers; patch tokens only
    def __init__(self, head_dim=64, K=16, theta=10000.0, scale=1.5, grid=14, clamp_like_official=False):
        super().__init__()
        nf, R = head_dim // 4, K // 2
        assert head_dim % (2 * K) == 0
        w = scale * theta ** (-torch.arange(nf).float() / nf)                 # omega_t
        d_idx = (torch.arange(nf) // 2) % R                                     # pair t -> direction (Eq. 13)
        self.register_buffer('w', torch.cat([w, w]))                            # (d/2,) per pair
        self.register_buffer('dir', torch.cat([d_idx, d_idx + R]))              # (d/2,) 0..K-1
        self.register_buffer('u', torch.stack([(torch.arange(K) * math.pi / K).cos(),
                                               (torch.arange(K) * math.pi / K).sin()], -1))  # (K,2)
        self.c, self.grid, self.clamp = (grid - 1) / 2, grid, clamp_like_official
    def forward(self, q, k, coords):                   # coords (N,2) integer (first axis = row); CLS row 0
        p = coords.float()
        t = p @ self.u.T if not self.clamp else ((p - self.c) @ self.u.T + self.c).clamp(0, self.grid - 1)
        ang = t[:, self.dir] * self.w                                           # (N, d/2)
        ang[0] = 0                                                              # CLS unrotated
        cos, sin = ang.cos().to(q.dtype), ang.sin().to(q.dtype)
        def apply(x):                                                           # (B,H,N,d), pairs (2c,2c+1)
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)
        return apply(q), apply(k)
```
(`u` is applied to `(row, col)`; the official grid_sample convention mirrors the angle sign, which only
re-labels directions within the same equally spaced set.)

---

## 5. LookHere (NeurIPS 2024, arXiv 2405.13985, github.com/GreenCUBIC/lookhere)

Sources: `lookhere.py` (`create_lh_bias_tensor`, `create_lh_layer_tensor`, `get_slopes`,
`get_none_slopes`, `LookHere.set_pos_embed`, `AttentionWithBiases`), paper §3 (Eq. 1–2), App. A.4/A.6,
and the `config` dicts inside the released checkpoints
`LH_{180,90,45}_weights_and_config.pth` [ckpt]:

    LH-180: {'global_slope': 2, 'layer_slopes': [1.5, 0.5], 'head_directions': 'halves-diagonals-none'}
    LH-90 : {'global_slope': 2, 'layer_slopes': [1.5, 0.5], 'head_directions': 'corners-triangles-none'}
    LH-45 : {'global_slope': 2, 'layer_slopes': [1.5, 0.5], 'head_directions': 'eighths1-eighths2-none'}

**Definition [paper Eq. 1–2, code].** Per layer l, head h, query i, key j (patch coords `(i_y, i_x)`):

    A^l = softmax(QK^T/√D_H − A_FIX^l)
    LookHere(l,h,i,j) = m(l,h) · Distance(i,j)   if j visible to i (inside head h's FOV)
                      = ∞                         otherwise
    Distance(i,j) = sqrt((i_y − j_y)^2 + (i_x − j_x)^2)      (Euclidean, patch units)
    m(l,h) = s_l(l) · s_h(h) · s_g

* `s_l(l) = linspace(1.5, 0.5, depth)[l]` (decreasing with depth) [code+paper].
* Directed heads: code `get_slopes` gives 0.5 and `global_slope = 2`, i.e. effective `s_h·s_g = 1`
  (paper: s_g = 1, directed heads unscaled). Undirected heads: code `get_none_slopes` =
  (1/4, 1/16, 1/64, 1/256) x 2 = **(1/2, 1/8, 1/32, 1/128)** [code = paper].
* Masked entries: `np.inf` → `nan_to_num(posinf=finfo(bfloat16).max)` → `× −1` → the bias tensor is
  stored in **bf16** with `−3.39e38` for masked cells (effectively −inf; softmax → 0). It is passed as a
  float `attn_mask` to `F.scaled_dot_product_attention` (or added to `q·k·scale`).
* **CLS:** bias row 0 and column 0 are zero (CLS attends everywhere unpenalised; every patch sees CLS
  with 0 bias). Self (i = j): visible in all FOVs (all masks are closed sets), bias 0.
* No learnable parameters, no APE (`del self.pos_embed`); the `(depth, H, N+1, N+1)` tensor is
  precomputed per resolution (`set_pos_embed(grid_size)`); only `s_l` differs across layers.

**FOVs (image coords: y = row index increases downward; dy = j_y − i_y, dx = j_x − i_x) [code].**
Masks are built as "cells to hide"; the visible sets are:
* half-planes (`halves`): up `dy ≤ 0`, down `dy ≥ 0`, left `dx ≤ 0`, right `dx ≥ 0` (180°).
* quadrants (`corners`): up-left `dy ≤ 0 & dx ≤ 0`, up-right, down-left, down-right (90°).
* cones (`triangles`): up `dy ≤ −|dx|`, down `dy ≥ |dx|`, left `dx ≤ −|dy|`, right `dx ≥ |dy|`
  (90°, bounded by the two diagonals through the query, inclusive).
* diagonal half-planes (`diagonals`): up∪right, up∪left, down∪right, down∪left cones (180°).
* 45° wedges: `eighths1` = (up-cone ∩ left), (down-cone ∩ left), (left-cone ∩ up), (right-cone ∩ up);
  `eighths2` = (up ∩ right), (down ∩ right), (left ∩ down), (right ∩ down).
* `none` = unmasked (distance penalty only); `zero` = no bias at all; `center` = distance to image centre.

**Head assignment [code] for H = 12 with a 3-part config `a-b-c`:** heads 0–3 ← a, 4–7 ← b, 8–11 ← c
(in the list order above). So LH-90 = 4 quadrant heads + 4 cone heads + 4 undirected (eight 90° views
covering 8 directions), LH-180 = 4 half-planes + 4 diagonal half-planes + 4 undirected, LH-45 = 8
wedges + 4 undirected. H = 16: `dirs[0]*2 + dirs[1] + dirs[2]` (8+4+4) with slopes
`get_slopes(8) = (0.5,0.5,0.5,0.5,0.25,0.25,0.25,0.25)` for the first group.

**H = 6 (DeiT-S) [code, analysed, not executed]:** the official branch is
`head_names = [a]*4 + [b]*2; head_directions = dirs[a] + dirs[b][:2]` (the third part is ignored). It
only runs when `b` is `none`/`zero` (e.g. `corners-none`: 4 quadrant heads + 2 undirected with
slopes 1/2, 1/8); for a directed `b` (e.g. `corners-triangles`) `get_slopes(2)` returns 4 slopes for 2
heads and the broadcast `(8,1,1) * (6,N,N)` raises. The paper gives no 6-head recipe. Two defensible
choices: (A) official-code-compatible `corners-none` / `halves-none` / `eighths1-none`
(4 directed + 2 undirected); (B) all-directed: first 6 FOVs of the 12-head list with `s_h = 1`
(what the code's intent appears to be). Report which one you use.

```python
def lookhere_bias(grid, heads, depth, variant='LH-90', six_head_mode='A',
                  s_g=2.0, s_l=(1.5, 0.5), neg=-torch.finfo(torch.bfloat16).max):
    ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')
    y, x = ys.flatten().float(), xs.flatten().float()
    dy, dx = y[None] - y[:, None], x[None] - x[:, None]                 # key minus query
    dist = (dy ** 2 + dx ** 2).sqrt()
    up, down, left, right = dy <= 0, dy >= 0, dx <= 0, dx >= 0          # 180 deg half-planes
    cu, cd, cl, cr = dy <= -dx.abs(), dy >= dx.abs(), dx <= -dy.abs(), dx >= dy.abs()   # 90 deg cones
    fov = {'LH-180': [up, down, left, right, cu | cr, cu | cl, cd | cr, cd | cl],
           'LH-90':  [up & left, up & right, down & left, down & right, cu, cd, cl, cr],
           'LH-45':  [cu & left, cd & left, cl & up, cr & up, cu & right, cd & right, cl & down, cr & down]}[variant]
    undirected = [0.25, 0.0625, 0.015625, 0.00390625]                   # x s_g=2 -> 1/2,1/8,1/32,1/128
    if heads == 12:   fovs, sh = fov, [0.5] * 8 + undirected
    elif heads == 16: fovs, sh = fov[:4] * 2 + fov[4:], [0.5] * 4 + [0.25] * 4 + [0.5] * 4 + undirected
    elif heads == 6:  fovs, sh = (fov[:4], [0.5] * 4 + undirected[:2]) if six_head_mode == 'A' else (fov[:6], [0.5] * 6)
    B = torch.zeros(depth, heads, grid * grid + 1, grid * grid + 1)
    for l, sl in enumerate(torch.linspace(s_l[0], s_l[1], depth)):
        for h in range(heads):
            b = -s_g * sl * sh[h] * dist
            if h < len(fovs): b = torch.where(fovs[h], b, torch.full_like(b, neg))
            B[l, h, 1:, 1:] = b                                          # CLS row/col stay 0
    return B                                                            # add to logits; (depth,H,N+1,N+1)
```

---

## 6. PaPE / PaPE-RI (arXiv 2602.01418, github.com/DTU-PAS/parabolic-position-encoding)

Sources: `pape/nn/positions/{pape_naive,pape_efficient,pape_ri}.py`, `pape/nn/encoder.py`,
`pape/nn/image_classifier.py`, `pape/configs.py`, paper §3–4, App. A.2/D, Table 10–11.

**Positions [code].** Images: `r = (col, row)` as **integers 1..n** (`arange(1, n+1)`), CLS gets
`r = (0, 0)` and is otherwise an ordinary token; `normalize=False` by default (option divides by
`(n_rows, n_cols)`). `p = num_positions = 2` for images.

**PaPE score [code+paper].** Per layer, with `x_i` = the LayerNorm'd block input (what `Attention`
receives as `hidden_state`), M parabolas per head:

    W_p : Linear(p → H·M, bias=False)     →  s_{h,i,ℓ} = w_{hℓ} · r_i          (projected positions)
    W_ab: Linear(d_model → 2·H·M, bias=False)
    a_{h,i,ℓ} = softplus((W_a x_i)_{hℓ}) ≥ 0,     b_{h,i,ℓ} = (W_b x_i)_{hℓ}
    S_ij = q_i·k_j − Σ_ℓ a_{iℓ} (s_jℓ − s_iℓ)^2 − Σ_ℓ b_{iℓ} (s_jℓ − s_iℓ)
    logits_ij = S_ij · D_head^{-1/2}       ← the positional terms ARE inside the 1/√D scaling [code]

(paper Eq.: `S_ij = ⟨a_i, Δr_ij^{⊙2}⟩ + ⟨b_i, Δr_ij⟩ + ⟨q_i,k_j⟩` with `a_i = −softplus(W_a x_i)`,
`Δr_ij = W_p(r_j − r_i)`; the sign of b is immaterial.) The bias is **content-dependent** (depends on
the query token), so it is a `(B, H, N, N)` tensor recomputed per forward, not a static buffer.
Official implementation realises it by augmenting q/k with `p² + 2p + 2 = 10` extra channels (naive:
`3M + 2`), padded to a multiple of 8, so FlashAttention/SDPA can be used; `self.scale = head_size^{-0.5}`
uses the *unpadded* head size.

**Init [code+paper].** PyTorch `nn.Linear` default (Kaiming-uniform) for `W_p`, `W_a`, `W_b`; paper:
"All models are randomly initialized with uniform Kaiming initialization". No special init.

**Defaults [code+paper].** `M = 50` (`PaPEConfig.parabolas`, tuned on CIFAR-10); per layer, per head;
no APE; ViT-B: head size 64, 12 heads, patch 16, 224px, 300 ep AdamW lr 6e-4, wd 0.05, batch 1024.
Params/layer = `2·H·M + d_model·2·H·M`: ViT-B 1,200 + 921,600 = 922,800 → 11.07 M total (paper:
11.1 M). DeiT-S (d_model 384, H 6): 600 + 230,400 = 231,000/layer → 2.77 M.

**PaPE-RI [code+paper].** Rotation-invariant restriction: `b = 0`, one shared `a` per head,
`W_p = w_p I`. Code: a **fixed, non-learned buffer** `s_h ~ U(−1/√p, 1/√p)` drawn once per layer per
head, `a_{h,i} = softplus((W_a x_i)_h)` with `W_a: Linear(d_model → H)`; bias
`= −a_{h,i} · s_h^2 · ‖r_j − r_i‖^2` (again inside the `D^{-1/2}` scale). Params/layer `d_model·H`:
ViT-B 9,216 → 110,592 total (paper: 110.6 K).

```python
class PaPE(nn.Module):           # returns bias (B,H,N,N); use logits = (q.k + bias) * head_dim**-0.5
    def __init__(self, d_model, heads, M=50, p=2):
        super().__init__()
        self.h, self.M = heads, M
        self.position = nn.Linear(p, heads * M, bias=False)                  # W_p
        self.ab = nn.Linear(d_model, 2 * heads * M, bias=False)              # [W_a; W_b]
    def forward(self, x_normed, coords):           # x_normed (B,N,d_model) = norm1(x); coords (N,2), CLS=(0,0)
        B, N, _ = x_normed.shape
        s = self.position(coords.float()).view(N, self.h, self.M).permute(1, 0, 2)     # (H,N,M)
        a, b = self.ab(x_normed).view(B, N, 2, self.h, self.M).permute(2, 0, 3, 1, 4)  # (B,H,N,M) each
        a = F.softplus(a)
        sT, s2T = s.transpose(-1, -2)[None], (s ** 2).transpose(-1, -2)[None]          # (1,H,M,N)
        quad = a @ s2T - 2 * (a * s[None]) @ sT + (a * (s ** 2)[None]).sum(-1, keepdim=True)  # Σ a (s_j - s_i)^2
        lin = b @ sT - (b * s[None]).sum(-1, keepdim=True)                             # Σ b (s_j - s_i)
        return -(quad + lin)                                                           # (B,H,N,N)

class PaPE_RI(nn.Module):
    def __init__(self, d_model, heads, p=2):
        super().__init__()
        self.register_buffer('s', (2 * torch.rand(heads) - 1) / math.sqrt(p))      # fixed per-head scale
        self.a = nn.Linear(d_model, heads, bias=False)
    def forward(self, x_normed, coords):
        a = F.softplus(self.a(x_normed)).transpose(1, 2)                            # (B,H,N)
        d2 = ((coords[None].float() - coords[:, None].float()) ** 2).sum(-1)       # (N,N)
        return -a[..., None] * (self.s ** 2)[None, :, None, None] * d2              # (B,H,N,N)
```

---

## 7. 2D-ALiBi (CROMA, NeurIPS 2023; as used by LookHere and the PaPE repo)

Sources: CROMA `pretrain_croma.py::get_alibi` / `use_croma.py::get_2dalibi` [code], CROMA paper §3
("biasing the self-attention matrix based on the Euclidean distance between query-key pairs ... we
calculate our attention-head slopes with the same geometric sequence as ALiBi") [paper],
PaPE `pape/nn/positions/alibi.py` ("nD-ALiBi", cites CROMA) [code], LookHere paper §2/§4.1/A.4.4.

**Definition [code].**

    bias[h, i, j] = − m_h · ‖(i_y, i_x) − (j_y, j_x)‖_2          (Euclidean, integer patch units; NOT Manhattan)
    m_h = ALiBi slopes (Press et al.): for H a power of 2, m_h = 2^{−8h/H}, h = 1..H;
          otherwise slopes(2^⌊log2 H⌋) ++ slopes(2·2^⌊log2 H⌋)[0::2][:H − 2^⌊log2 H⌋]
    H = 12: (2^-1, ..., 2^-8, 2^-0.5, 2^-1.5, 2^-2.5, 2^-3.5)
    H = 6 : (2^-2, 2^-4, 2^-6, 2^-8, 2^-1, 2^-3)
    H = 16: 2^-0.5, 2^-1, ..., 2^-8

Fixed, identical in every layer, added to `q·k·d^-0.5` before softmax. CROMA has no CLS token
(mean pooling). CLS handling therefore differs between the users of this baseline:
* PaPE repo: CLS is an ordinary token at `r = (0,0)` with patches at 1..n, so the CLS row/column carry
  real distance penalties.
* LookHere (paper: "2D-ALiBi ... is equivalent to LookHere without our 2D directional masks"; its code
  config `none`): CLS row/column = 0, `s_l` layer schedule 1.5→0.5 and slopes
  `get_none_slopes(12)·2 = (1/2,1/2,1/2, 1/8,1/8,1/8, 1/32,...,1/128,...)` [inferred from the code's
  `none` option; the exact config used for the paper's 2D-ALiBi row is not released], global slope
  tuned on minival (1.0 at 224px, up to 1.6 at 1024px; paper Table 9).

```python
def alibi_slopes(H):
    def pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3))); return [start * start ** i for i in range(n)]
    if math.log2(H).is_integer(): return pow2(H)
    c = 2 ** math.floor(math.log2(H))
    return pow2(c) + alibi_slopes(2 * c)[0::2][:H - c]

def alibi2d_bias(grid, H, cls='zero'):                 # -> (H, N+1, N+1), add to scaled logits
    ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')
    pts = torch.stack([ys.flatten(), xs.flatten()], -1).float()
    if cls == 'pape':                                  # PaPE repo: CLS at (0,0), patches at 1..n
        pts = torch.cat([torch.zeros(1, 2), pts + 1])
    bias = -torch.tensor(alibi_slopes(H)).view(H, 1, 1) * torch.cdist(pts, pts)
    if cls == 'zero':                                  # CROMA/LookHere style: CLS row & col = 0
        bias = F.pad(bias, (1, 0, 1, 0))
    return bias
```

---

## 8. iRPE (ICCV 2021, arXiv 2107.14222, github.com/microsoft/Cream/tree/main/iRPE)

Sources: `DeiT-with-iRPE/irpe.py` (`piecewise_index`, `_rp_2d_product`, `get_bucket_ids_2d`, `iRPE`,
`get_rpe_config`, `build_rpe`), `rpe_vision_transformer.py` (`RPEAttention`, `VisionTransformer`),
`rpe_models.py::deit_small_patch16_224_ctx_product_50_shared_k`, `main.py` (timm `create_optimizer`),
README (DeiT-S iRPE-K 80.9 top-1). Implementation: `pe_extra.py::IRPE`, name `irpe`.

**Config [code].** `get_rpe_config(ratio=1.9, method="product", mode="ctx", shared_head=True, skip=1,
rpe_on="k")`: alpha, beta, gamma = 1.9, 3.8, 15.2. The piecewise index is

    g(x) = round(x)                                                               |x| <= alpha
         = sign(x) * min(beta, round(alpha + ln(|x|/alpha) / ln(gamma/alpha) * (beta - alpha)))   otherwise

followed by a cast to long. The float clip at beta = 3.8 truncates to 3, so g ∈ [-3, 3]
(g(0..30) = 0,1,2,2,3,3,...). Product method: bucket = (g(Δrow)+3)·7 + (g(Δcol)+3) ∈ [0,49), with
Δ = p_query − p_key. One extra `skip` bucket (id 49) fills every CLS row and column (CLS–CLS too), giving
50 buckets. Contextual mode on keys: `attn = (q·s)k^T + (q·s) r_bucket`, where s = d^-0.5 and
r = `lookup_table_weight` (1, d, 50). Tables are shared over heads, there is one per layer (every
RPEAttention builds its own), and they are zero-initialised (default initializer). The released DeiT models keep the
learned `pos_embed`, so `ape=True` by default. `ape=False` is available as an ablation.

**Resolution.** g saturates at |Δ| >= 4, so every grid maps onto the same 49 buckets. No resampling is
needed, and this matches the official code at any height/width.

**Implementation.** Uses the `ctx.attn_fn` hook: lut = (q·s) @ W (B,H,N,50), then `gather` with the cached
(N,N) bucket ids (the same algorithm as the official non-`rpe_ops` path). Params:
12·64·50 = 38,400 (+ 75,264 APE).

**Deviations.** None in the attention math. Under this repo's train.py the 3D table is weight-decayed, as
with timm `create_optimizer` in the official code. The APE is decayed too, following this repo's
convention for every learned-APE baseline. Official DeiT exempts `pos_embed`.

---

## 9. CrossFormer Dynamic Position Bias (ICLR 2022, arXiv 2108.00154, github.com/cheerss/CrossFormer)

Sources: `models/crossformer.py` (`DynamicPosBias`, `Attention.__init__/forward`, `_init_weights`).
Implementation: `pe_extra.py::DPB`, name `dpb`.

**Config [code].** Each attention layer builds `DynamicPosBias(dim // 4, num_heads, residual=False)`.
The only call site uses `residual=False`, so that is the default here; the residual variant is kept as a kwarg.
The module's `pos_dim = (dim // 4) // 4`, i.e. 24 for DeiT-S. The MLP is
`Linear(2, 24) → [LN, ReLU, Linear(24,24)] ×2 → LN, ReLU, Linear(24, H)`. Its input is the integer
offset (Δrow, Δcol) = p_query − p_key (`relative_coords`), unnormalised. It is evaluated on the
(2h−1)(2w−1) offsets of the current grid and gathered with `relative_position_index`. Init:
trunc_normal(0.02) weights, zero biases, LN (1, 0). No APE.

**Resolution.** An MLP of the offset, so any grid works. Offsets beyond ±13 are MLP extrapolation.

**Deviations / assumptions.**
1. CrossFormer applies DPB inside 7×7 groups (SDA) or dilated groups (LDA) and has no CLS token. Here it runs
   over the global 14×14 attention of DeiT-S, with a CLS row/column bias of 0 [assumption, same as ALiBi/LookHere].
2. Consequence: in the last block only the CLS row reaches the classifier and it carries no DPB term, so the
   layer-11 DPB MLP gets **zero gradient** (1,566 params stay at init). This is inherent to CLS=0, not a bug.
   Params: 12 × 1,566 = 18,792, of which 17,226 are trained.
3. The MLP runs in fp32 (`_fp32()`).

---

## 10. ConViT GPSA (ICML 2021, arXiv 2103.10697, github.com/facebookresearch/convit)

Sources: `convit.py` (`GPSA.get_attention`, `local_init`, `get_rel_indices`, `VisionTransformer`
with `local_up_to_layer=10`, `locality_strength=1.`, `use_pos_embed=True`), `models.py` (4/9/16 heads).
Implementation: `pe_extra.py::GPSA`, name `gpsa`.

**Formula [code].** In blocks 0..9:

    A = (1 − σ(λ_h)) softmax(q k^T d^-0.5) + σ(λ_h) softmax(pos_proj(r_ij)),   r_ij = (Δx, Δy, Δx²+Δy²)

Here Δ = p_key − p_query in integer patch units, `pos_proj = Linear(3, H)` with bias, and λ_h is
initialised to 1. Blocks 10 and 11 are plain MHSA. A learned APE is added to the patch tokens.

**Locality init [code].** kernel = int(sqrt(H)) and c = (kernel−1)/2 for an even kernel. For head
h1 + kernel·h2 the weights are v_pos = (2(h2−c), 2(h1−c), −1) × locality_strength, which gives logits
−|Δ − c_h|² + const. Other weights are trunc_normal(0.02) and the bias is 0. **With H = 6 (DeiT-S):** kernel = 2, so heads 0–3 get the local init
(centres (±0.5, ±0.5)) and heads 4–5 keep the random init. This is exactly what the official code does for
6 heads. ConViT's own models use square head counts and never hit this case.

**Resolution.** r_ij is computed for the current grid (official code assumes a square grid; non-square is
supported here). Positional logits are −|Δ|²-like, so attention stays local at any grid.

**Deviations.**
1. **CLS:** ConViT concatenates CLS only *after* the GPSA blocks, but our backbone has it from block 0.
   CLS query rows therefore use content attention only. For patch queries the positional softmax runs over
   patch keys only: the CLS key gets positional weight 0, so its total weight is (1−σ)·content.
2. GPSA `local_init` also sets W_v = I. Here W_v is part of timm's fused qkv and keeps the timm init.
   (Reproducing it would need a backbone-init hook in vit.py, which was deliberately not added.)
3. The official `attn /= attn.sum(-1)` is omitted because rows already sum to 1 (checked numerically).
4. Positional scores and gates are computed in fp32.
5. λ is stored as one 1D parameter per layer, so the repo's ndim<=1 rule exempts it from weight decay
   (as timm does for `gating_param`).

Params: 10 × (3·6 + 6 + 6) = 300 (+ 75,264 APE).

---

## 11. Wavelet-based Positional Representation, 2D port (Oka et al., ICLR 2025, arXiv 2502.02004)

Sources: the paper (HTML v1), §5.1 Eqs. (10)–(14), §6.1/App. A.7 (short context), §7.1 (long context),
App. A.6 (implementation tips), A.13 Table 3 (scale/shift ablation). **No official code is linked.**
Implementation: `pe_extra.py::WaveletRPE`, name `wavelet_rpe`.

**1D method [paper].**

    e_mn = (q_m k_n^T + q_m p_mn^T) / sqrt(d)                                   (Eq. 10)
    p_mn[c] = ψ((t − b_c)/a_c),  t = m − n,  ψ(t) = (1 − t²) exp(−t²/2)         (Eqs. 11, 12, 14)
    (a, b) ∈ {2^0, ..., 2^(s−1)} × {0, ..., d/s − 1}                            (Eq. 13)

This gives one wavelet per head channel (d pairs). All amplitudes are equal (no 1/sqrt(a)). There is no
clipping ("the distance of the position expression is fixed regardless of the length") and there are no
parameters. The bank depends only on d, so it is identical across heads (and, implicitly, layers).
Paper settings: d = 128, s = 8 (a = 1..128) × 16 shifts at L_train = 512, and a ∈ {2^2..2^9} for Llama-2
long context. Table 3: more scales help, while trading scales for shifts or using a single scale hurts.
The paper sets no APE. The mechanism is RPE-style (Shaw): the query dotted with a relative embedding, on
keys only.

**2D port [ours].**
* **mode="axial" (default).** Head channels 0..31 use Δrow and 32..63 use Δcol (the axial-RoPE channel
  split), each half carrying the paper's 1D bank. Then q·p_ij = q_row·ψ(Δrow_ij) + q_col·ψ(Δcol_ij).
  Why not the Euclidean radius: the paper's t is signed and the causal mask supplies direction. A
  bidirectional 2D encoder has no mask, and a radial bank makes the positional term direction-blind
  (iRPE found directed > undirected encodings for ViTs).
* **Scales** {1, 2, 4, 8} (s = 4). 2^4 = 16 already exceeds the largest offset (13) of the 14×14 training grid,
  mirroring the paper's largest scale 128 ≈ L_train/4. 32 channels / 4 scales gives 8 shifts per axis.
* **Shifts.** The paper's set {0..n−1} covers the causal half-line. For a signed axis we use the same
  half-line on both sides: b ∈ {0,1,2,3} ∪ {−0,−1,−2,−3}. The set is reflection-symmetric, so horizontal
  flips map the bank onto itself. b = ±0 duplicates one channel per scale, which is harmless.
* Δ = p_query − p_key in patch units, never rescaled with the grid (as in the paper). At larger test grids,
  distant pairs therefore fall back to content-only logits (ψ → 0). `pos_scale` multiplies Δ.
* The (a, b) → channel order is unspecified in the paper. We use scale-major order; since q has no other
  transform, a permutation is absorbed by W_Q.
* CLS row/column: no positional term (the paper has no CLS). No APE. Same bank in every head and layer.
  0 parameters.
* **mode="radial" (ablation).** ψ((‖Δ‖₂ − b)/a) with the paper's literal 4 scales × 16 shifts {0..15}.

**Implementation.** Uses the `ctx.attn_fn` hook (the bias needs q). The method follows App. A.6 Tip 2:
it precomputes the (values × d) wavelet table, multiplies by q, and gathers. In axial mode the bias is
additively separable, so the gathers are (B,H,n,h) and (B,H,n,w), followed by one broadcast add.

---

## Attention hook added for §8–§11 (`ctx.attn_fn`)

`PEContext.attn_fn: (logits, q, layer) -> probabilities` (default None, reset in `clear()`).
`logits = q·k·d^-0.5 (+ attn_bias + bias_fn)`, and `q` is the unscaled full-sequence query after any
rotation or `qk_fn`. When it is set, PEAttention takes the explicit-attention path and calls it in place of
`softmax`. Dropout, `keep_attn`, value/LePE hooks and the projection are unchanged. With `attn_fn=None` the
code path is byte-for-byte the old one. Verified bit-identical outputs and gradients (CPU, fp32, seed 0,
224 and 96 px) for none, rope_mixed, lepe, lookhere, viper, pape, liere, alibi, relbias. relbias@224
gradients were already run-to-run nondeterministic before the change (index accumulation).

---

## Open items you should decide (could not be settled from the sources)

1. **STRING**: init of S / circulant scalars, dense vs sparse S, per-layer sharing, coordinate units,
   CLS, APE — all unspecified in the paper; no official code. The spec above states my assumptions.
2. **LieRE block size**: paper headline = 8, repo default = 64. Run 8 (and optionally 64).
3. **Spiral RoPE K**: paper = 16 for DeiT; repo script default = 4. Use 16. Decide whether to copy the
   official `grid_sample` border clamping (`clamp_like_official=True`) or the paper's exact `p·u_k`.
4. **ComRoPE**: `init_std` (code default 1.0) is not stated in the paper; paper setting removes the APE
   while the code default keeps one (inverted `keep_ape` flag). CLS at (0.5, 0.5).
5. **LookHere with 6 heads**: official code only supports `<directed>-none` (4 + 2); pick mode A or B.
6. **2D-ALiBi CLS**: zero row/col (CROMA/LookHere) vs CLS at (0,0) (PaPE repo).
7. **PaPE**: remember the bias is scaled by `head_dim^-0.5` together with `q·k`, and is computed from the
   pre-attention LayerNorm output.
8. **iRPE**: APE kept (released models) — `ape=False` if an RPE-only row is wanted.
9. **DPB**: CLS bias 0 leaves the last layer's MLP untrained; alternative would be an extra learned CLS bias.
10. **GPSA**: 6 heads → only 4 locality-initialised heads (official behaviour); W_v = I init not reproduced.
11. **Wavelet RPE**: axial vs radial 2D port, scale set {1,2,4,8}, mirrored shifts — all our choices (§11).

All sketches above were executed (`scratchpad/scripts/selfcheck.py`): shapes, norm preservation
(orthogonality), exact relative property for ComRoPE-AP/LD, Circulant FFT == dense `exp(C − Cᵀ)`,
Spiral direction/frequency pairing for K ∈ {2,4,16} (K=2 == axial RoPE), LookHere FOV coverage and
zero CLS row/col, PaPE matmul form == naive formula == official q/k augmentation, ALiBi slopes.

All sources are the public papers and official repositories cited in each section.
