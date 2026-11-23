
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .integrated_gradients import (
    integrated_gradients,
    target_prediction,
)
from .faithfulness import score_blocks


@dataclass
class RobustnessResult:
    cosine_similarity: torch.Tensor
    relative_l1_change: torch.Tensor
    topk_overlap: torch.Tensor
    prediction_shift: torch.Tensor


def attribution_robustness(
    model,
    inputs,
    perturbed_inputs,
    original_ig,
    baseline,
    *,
    horizon,
    target_channel=0,
    target_step=-1,
    ig_steps=32,
    block_size=10,
    top_fraction=0.2,
):
    """
    So sánh attribution trước và sau khi gây nhiễu.

    inputs:          (B, L, C)
    perturbed_inputs:(B, L, C)
    original_ig:     (B, L, C), từ Phase 1
    """

    if model.training:
        raise ValueError("Model phải ở chế độ eval()")

    if (
        inputs.ndim != 3
        or inputs.shape != perturbed_inputs.shape
        or inputs.shape != original_ig.shape
    ):
        raise ValueError("Input và attribution không khớp shape")

    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction không hợp lệ")

    if not torch.isfinite(perturbed_inputs).all():
        raise ValueError("Perturbed inputs chứa NaN/Inf")

    # 1. Tính lại IG sau khi thêm nhiễu
    noisy_result = integrated_gradients(
        model=model,
        inputs=perturbed_inputs,
        baseline=baseline,
        horizon=horizon,
        target_channel=target_channel,
        target_step=target_step,
        steps=ig_steps,
    )

    old_ig = original_ig.detach().cpu()
    new_ig = noisy_result.attribution

    old_flat = old_ig.flatten(1)
    new_flat = new_ig.flatten(1)

    # 2. Cosine similarity trên từng sample
    old_norm = old_flat.norm(dim=1)
    new_norm = new_flat.norm(dim=1)

    if (old_norm < 1e-10).any() or (new_norm < 1e-10).any():
        raise ValueError(
            "Attribution gần zero; cosine không xác định"
        )

    cosine = F.cosine_similarity(
        old_flat, new_flat, dim=1
    )

    # 3. Relative L1 attribution change
    relative_l1 = (
        (new_flat - old_flat).abs().sum(dim=1)
        / old_flat.abs().sum(dim=1).clamp_min(1e-10)
    )

    # 4. So sánh top-K block-channel
    old_scores = score_blocks(
        old_ig, block_size
    ).flatten(1)

    new_scores = score_blocks(
        new_ig, block_size
    ).flatten(1)

    total_units = old_scores.shape[1]
    k = max(1, math.ceil(total_units * top_fraction))

    old_top = torch.topk(old_scores, k, dim=1).indices
    new_top = torch.topk(new_scores, k, dim=1).indices

    overlap = (
        old_top.unsqueeze(2) == new_top.unsqueeze(1)
    ).any(dim=2).float().mean(dim=1)

    # 5. Độ thay đổi của dự báo gốc
    with torch.no_grad():
        original_prediction = target_prediction(
            model(inputs),
            channels=inputs.shape[-1],
            horizon=horizon,
            target_channel=target_channel,
            target_step=target_step,
        )

        perturbed_prediction = target_prediction(
            model(perturbed_inputs),
            channels=inputs.shape[-1],
            horizon=horizon,
            target_channel=target_channel,
            target_step=target_step,
        )

    shift = (
        original_prediction - perturbed_prediction
    ).abs().cpu()

    return RobustnessResult(
        cosine_similarity=cosine.cpu(),
        relative_l1_change=relative_l1.cpu(),
        topk_overlap=overlap.cpu(),
        prediction_shift=shift,
    )