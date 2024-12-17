import torch
import triton

from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.utils import (
    amp_custom_bwd,
    amp_custom_fwd,
    element_mul_kernel,
    is_hip,
)

MAX_FUSED_SIZE = 65536 // 2


def weighted_fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    sample_weights=None,  # New parameter for per-sample weights
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
):
    device = _input.device
    dtype = _input.dtype
    BT, H = _input.shape
    V = weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    num_chunks = triton.cdiv(BT, chunk_size)

    grad_weight = (
        torch.zeros_like(weight, device=device) if weight.requires_grad else None
    )
    grad_input = torch.zeros_like(_input, device=device)
    grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)

    # Handle sample weights
    if sample_weights is None:
        sample_weights = torch.ones(BT, dtype=dtype, device=device)
    else:
        sample_weights = sample_weights.to(dtype=dtype)

    # Calculate total weight for non-ignored samples
    valid_mask = target != ignore_index
    weighted_valid_mask = valid_mask * sample_weights
    total_weight = weighted_valid_mask.sum()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]
        weights_chunk = sample_weights[start_idx:end_idx]

        logits_chunk = _input_chunk @ weight.t()
        if bias is not None:
            logits_chunk = logits_chunk + bias
        target_chunk = target[start_idx:end_idx]

        n_rows = logits_chunk.shape[0]
        loss_1d_slice = loss_1d[start_idx:end_idx]

        # Calculate weighted non-ignore count for this chunk
        chunk_valid_mask = target_chunk != ignore_index
        chunk_weights = weights_chunk * chunk_valid_mask
        n_weighted_non_ignore = chunk_weights.sum().item()

        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),
            loss_ptr=loss_1d_slice,
            z_loss_ptr=loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1),
            n_cols=V,
            n_non_ignore=n_weighted_non_ignore,  # Pass weighted count
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap if softcap is not None else 0.0,
            RETURN_Z_LOSS=0,
            HAS_SOFTCAPPING=True if softcap is not None else False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        if reduction == "mean":
            alpha = n_weighted_non_ignore / total_weight if total_weight > 0 else 0.0
        else:
            alpha = 1.0

        # Apply sample weights to loss and gradients
        weights_chunk = weights_chunk.view(-1, 1)
        loss_1d[start_idx:end_idx] = loss_1d_slice * alpha * weights_chunk.squeeze()
        grad_logits_chunk = logits_chunk * alpha * weights_chunk

        grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

        if grad_weight is not None:
            # Apply weights to gradient computation
            weighted_input_chunk = _input_chunk * weights_chunk
            torch.addmm(
                input=grad_weight,
                mat1=logits_chunk.t(),
                mat2=weighted_input_chunk,
                out=grad_weight,
                alpha=alpha,
                beta=1.0,
            )

        if bias is not None:
            # Apply weights to bias gradient computation
            weighted_logits_sum = (logits_chunk * weights_chunk).sum(dim=0)
            torch.add(
                input=grad_bias,
                other=weighted_logits_sum,
                out=grad_bias,
                alpha=alpha,
            )

    loss = torch.sum(loss_1d)
    return loss, grad_input, grad_weight, grad_bias


def weighted_fused_linear_cross_entropy_backward(
    grad_output, grad_input, grad_weight, grad_bias, sample_weights=None
):
    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    if torch.ne(grad_output, torch.tensor(1.0, device=grad_output.device)):
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        # Apply sample weights if provided
        if sample_weights is not None:
            grad_output = grad_output * sample_weights

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        # handle grad_weight
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
        sample_weights=None,  # New parameter
        bias=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
    ):
        loss, grad_input, grad_weight, grad_bias = (
            weighted_fused_linear_cross_entropy_forward(
                _input,
                weight,
                target,
                sample_weights,  # Pass sample weights
                bias,
                ignore_index,
                lse_square_scale,
                label_smoothing,
                reduction,
                softcap,
            )
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
        grad_input, grad_weight, grad_bias = (
            weighted_fused_linear_cross_entropy_backward(
                grad_output, grad_input, grad_weight, grad_bias
            )
        )
        # Add None for sample_weights gradient
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
