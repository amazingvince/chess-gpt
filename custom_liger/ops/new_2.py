import torch
import triton

from custom_liger.ops.weighted_liger_cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.utils import (
    amp_custom_bwd,
    amp_custom_fwd,
    element_mul_kernel,
    is_hip,
)

MAX_FUSED_SIZE = 65536 // 2


def fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    sample_weights=None,
):
    if sample_weights is None:
        raise ValueError("sample_weights must be provided for weighted version.")
    device = _input.device

    BT, H = _input.shape
    V = weight.shape[0]
    B = sample_weights.shape[0]
    T = BT // B
    if B * T != BT:
        raise ValueError("sample_weights length does not match input batch size.")

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    inc_factor = (V + H - 1) // H
    from math import ceil, log2

    chunk_size = 2 ** (int(log2((BT + inc_factor - 1) // inc_factor)))
    num_chunks = (BT + chunk_size - 1) // chunk_size

    grad_weight = (
        torch.zeros_like(weight, device=device) if weight.requires_grad else None
    )
    grad_input = torch.zeros_like(_input, device=device)
    grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    total_n_non_ignore = (target != ignore_index).sum()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]

        logits_chunk = _input_chunk @ weight.t()
        if bias is not None:
            logits_chunk = logits_chunk + bias
        target_chunk = target[start_idx:end_idx]
        n_rows = logits_chunk.shape[0]
        loss_1d_slice = loss_1d[start_idx:end_idx]
        n_non_ignore = (target_chunk != ignore_index).sum().item()

        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        # Call kernel with sample_weights and T
        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),
            loss_ptr=loss_1d_slice,
            z_loss_ptr=loss_1d_slice,  # dummy
            loss_stride=loss_1d_slice.stride(-1),
            n_cols=V,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap if softcap is not None else 0.0,
            RETURN_Z_LOSS=0,
            HAS_SOFTCAPPING=True if softcap is not None else False,
            BLOCK_SIZE=BLOCK_SIZE,
            sample_weights_ptr=sample_weights,
            T=T,
            num_warps=32 if not is_hip() else 16,
        )

        # Scaling factor for chunk (same logic as original)
        if reduction == "mean":
            alpha = n_non_ignore / total_n_non_ignore if total_n_non_ignore > 0 else 0.0
        else:
            alpha = 1.0

        loss_1d[start_idx:end_idx] = loss_1d_slice * alpha
        grad_logits_chunk = (
            logits_chunk * alpha
        )  # already scaled in kernel by sample_weight

        grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

        if grad_weight is not None:
            # h(∇xL)ᵀ = logits_chunkᵀ * input_chunk for W gradient
            # Note: grad_logits_chunk already scaled by alpha and sample_weight in kernel.
            grad_weight.addmm_(grad_logits_chunk.t(), _input_chunk, alpha=1.0)

        if bias is not None:
            grad_bias.add_(grad_logits_chunk.sum(dim=0), alpha=1.0)

    loss = torch.sum(loss_1d)
    return loss, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(
    grad_output, grad_input, grad_weight, grad_bias
):
    if torch.ne(grad_output, torch.tensor(1.0, device=grad_output.device)):
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V
            element_mul_kernel[(n_rows,)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )

        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V
            element_mul_kernel[(n_rows,)](
                grad_bias,
                grad_bias.stride(-1),
                grad_output,
                1,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )
    return grad_input, grad_weight, grad_bias


class WeightedLigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        sample_weights,
        bias=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
    ):
        loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
            _input,
            weight,
            target,
            bias,
            ignore_index,
            lse_square_scale,
            label_smoothing,
            reduction,
            softcap,
            sample_weights=sample_weights,
        )
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if bias is not None else None,
        )
        return loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        return (
            grad_input,
            grad_weight,
            None,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )
