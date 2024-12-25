import logging

from typing import Callable

import transformers

from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.transformers.swiglu import (
    LigerSwiGLUMLP,
)
from packaging import version
from transformers import PreTrainedModel


transformer_version = version.parse(transformers.__version__)

logger = logging.getLogger(__name__)
SUPPORTED_TRANSFORMER_VERSION = "4.46.1"
TRANSFORMER_DEPRECATION_WARNING = "Support for transformers versions < 4.46.1 will soon be discontinued due to issues with incorrect gradient accumulation. \n Please consider upgrading to avoid potential issues. See details: https://github.com/huggingface/transformers/pull/34191"


def _bind_method_to_module(module, method_name: str, new_method: Callable):
    # Binds a new method to a module instance so that self is passed as the first argument
    module.__dict__[method_name] = new_method.__get__(module, module.__class__)


def _patch_rms_norm_module(
    module, offset=0.0, eps=1e-6, casting_mode="llama", in_place=True
):
    module.offset = offset
    module.casting_mode = casting_mode
    module.variance_epsilon = (
        getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
    )
    module.in_place = in_place
    _bind_method_to_module(module, "forward", LigerRMSNorm.forward)
    _bind_method_to_module(module, "extra_repr", LigerRMSNorm.extra_repr)


def apply_liger_kernel_to_chess_llama(
    rope: bool = True,
    cross_entropy: bool = False,
    weighted_fused_linear_cross_entropy: bool = True,
    fused_linear_cross_entropy: bool = False,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Llama models (2 and 3)

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """

    assert (
        sum(
            [
                cross_entropy,
                fused_linear_cross_entropy,
                weighted_fused_linear_cross_entropy,
            ]
        )
        <= 1
    ), "Only one of cross_entropy, fused_linear_cross_entropy, or weighted_fused_linear_cross_entropy can be True."

    from chess_gpt.model import chess_llama_2 as chess_llama
    from chess_gpt.model.chess_llama_2 import LlamaModel

    if rope:
        chess_llama.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        chess_llama.LlamaRMSNorm = LigerRMSNorm
    if swiglu:
        chess_llama.LlamaMLP = LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules (e.g. LlamaRMSNorm or LlamaMLP)

        # get the base model from the model instance
        base_model: LlamaModel = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _bind_method_to_module(
                    decoder_layer.mlp, "forward", LigerSwiGLUMLP.forward
                )
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)

    print("Liger kernels applied to Chess Llama model.")
