"""
ViPER: Vision Positional Encoding with Multi-scale Dynamic Wavelet Encoding.

Public API:
    from viper import (
        ViPERConfig, ViPERFeatureExtractor,
        DeiTWithCustomPE, make_pe_dynamic,
        train_one, train_one_multires,
        evaluate, build_model, set_seed,
    )
"""

from .viper import ViPERConfig, ViPERFeatureExtractor, ChannelProjection

from .pe_methods import (
    NoPE, LearnedPE, SinCos2DPE, Relative2DPE, CPE, PerBlockPEG,
)

from .extra_pes import (
    RoPEMixed2D, ALiBi2D, CustomAttention, apply_internal_pe_to_deit,
)

from .model import DeiTWithCustomPE, make_pe_dynamic

from .trainer import (
    set_seed, compute_metrics, evaluate,
    build_model, train_one, train_one_multires,
    load_data_loader_module,
    INTERNAL_PE_TYPES, MULTIRES_SUPPORTED,
)

__all__ = [
    "ViPERConfig", "ViPERFeatureExtractor", "ChannelProjection",
    "NoPE", "LearnedPE", "SinCos2DPE", "Relative2DPE", "CPE", "PerBlockPEG",
    "RoPEMixed2D", "ALiBi2D", "CustomAttention", "apply_internal_pe_to_deit",
    "DeiTWithCustomPE", "make_pe_dynamic",
    "set_seed", "compute_metrics", "evaluate",
    "build_model", "train_one", "train_one_multires",
    "load_data_loader_module",
    "INTERNAL_PE_TYPES", "MULTIRES_SUPPORTED",
]
