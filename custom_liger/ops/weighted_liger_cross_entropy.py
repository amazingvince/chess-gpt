import operator
from typing import Optional

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import compare_version, element_mul_kernel, is_hip

if compare_version("triton", operator.ge, "3.0.0"):
    try:
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh

_TRUE = tl.constexpr(1)
_FALSE = tl.constexpr(0)


@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    z_loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    softcap,
    RETURN_Z_LOSS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
    # New arguments
    sample_weights_ptr,  # pointer to sample_weights of shape [B]
    T: tl.int32,  # length of each sample (BT is total, so B = BT/T)
):
    program_id = tl.program_id(0).to(tl.int64)

    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    X_ptr += program_id * X_stride

    # Identify sample index and load sample weight
    sample_idx = program_id // T
    sample_weight = tl.load(sample_weights_ptr + sample_idx)

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride
    z_loss_ptr += program_id * loss_stride

    # First pass: find max + sum
    m = float("-inf")
    d = 0.0
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)
    if HAS_SOFTCAPPING:
        ori_X_y = softcap * tanh(ori_X_y / softcap)

    eps = label_smoothing / n_cols
    scaled_x_sum = 0.0

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).cast(tl.float32)
        if HAS_SOFTCAPPING:
            X_block = softcap * tanh(X_block / softcap)
        block_max = tl.max(X_block)
        if label_smoothing > 0:
            scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    lse = m + tl.log(d)

    # Second pass: compute gradients
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).cast(tl.float32)
        if HAS_SOFTCAPPING:
            intermediate = tanh(X_block / softcap)
            X_block = softcap * intermediate
        X_block = tl.exp(X_block - m) / d
        X_block += 2 * lse_square_scale * lse * X_block
        X_block += -eps
        X_block = tl.where(X_offsets != y, X_block, X_block - (1 - label_smoothing))
        if reduction == "mean":
            X_block = X_block / n_non_ignore
        if HAS_SOFTCAPPING:
            X_block = X_block * (1 - intermediate * intermediate)

        # Scale gradients by sample_weight
        X_block = X_block * sample_weight

        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    tl.debug_barrier()

    # Compute the loss
    loss = lse - ori_X_y
    if label_smoothing > 0:
        smooth_loss = scaled_x_sum + label_smoothing * lse
        loss = loss * (1 - label_smoothing) + smooth_loss

    z_loss = lse_square_scale * lse * lse
    loss += z_loss

    if reduction == "mean":
        z_loss = z_loss / n_non_ignore
        loss = loss / n_non_ignore

    # Scale loss by sample_weight
    loss = loss * sample_weight
    z_loss = z_loss * sample_weight

    tl.store(loss_ptr, loss)
    if RETURN_Z_LOSS == _TRUE:
        tl.store(z_loss_ptr, z_loss)


def cross_entropy_forward(
    _input,
    target,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    softcap,
    return_z_loss,
    sample_weights=None,
):
    if sample_weights is None:
        raise ValueError("sample_weights must be provided for weighted version.")
    BT, V = _input.shape
    B = sample_weights.shape[0]
    T = BT // B
    if B * T != BT:
        raise ValueError("sample_weights length does not match input batch size.")

    if not isinstance(return_z_loss, int):
        return_z_loss = _TRUE.value if return_z_loss else _FALSE.value

    BLOCK_SIZE = min(65536 // 2, triton.next_power_of_2(V))

    loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device)
    if return_z_loss == _TRUE.value:
        z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device)
    else:
        z_loss_1d = loss_1d  # dummy

    n_non_ignore = (target != ignore_index).sum().item()

    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    liger_cross_entropy_kernel[(BT,)](
        X_ptr=_input,
        X_stride=_input.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),
        loss_ptr=loss_1d,
        z_loss_ptr=z_loss_1d,
        loss_stride=loss_1d.stride(-1),
        n_cols=V,
        n_non_ignore=n_non_ignore,
        ignore_index=ignore_index,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        reduction=reduction,
        softcap=softcap if softcap is not None else 0.0,
        RETURN_Z_LOSS=return_z_loss,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_SOFTCAPPING=True if softcap is not None else False,
        sample_weights_ptr=sample_weights,
        T=T,
        num_warps=32 if not is_hip() else 16,
    )

    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss == _TRUE.value else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss == _TRUE.value else None

    return loss, z_loss, _input


def cross_entropy_backward(_input, grad_output):
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        pass
    else:
        BT, V = _input.shape
        BLOCK_SIZE = min(65536 // 2, triton.next_power_of_2(V))
        element_mul_kernel[(BT,)](
            _input,
            _input.stride(-2),
            grad_output,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )
    return _input


class WeightedLigerCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _input: torch.Tensor,
        target: torch.Tensor,
        sample_weights: torch.Tensor,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
    ):
        loss, z_loss, _input = cross_entropy_forward(
            _input,
            target,
            ignore_index,
            lse_square_scale,
            label_smoothing,
            reduction,
            softcap,
            return_z_loss,
            sample_weights=sample_weights,
        )
        ctx.save_for_backward(_input.detach())
        ctx.return_z_loss = return_z_loss
        return loss, z_loss

    @staticmethod
    def backward(ctx, grad_output, grad_output2):
        if ctx.return_z_loss:
            del grad_output2
        (_input,) = ctx.saved_tensors
        _input = cross_entropy_backward(_input, grad_output)
        return _input, None, None, None, None, None, None, None, None
