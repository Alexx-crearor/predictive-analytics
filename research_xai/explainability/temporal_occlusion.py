"""Occlusion theo block thời gian và theo từng channel, không cần gradient."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .integrated_gradients import target_prediction


@dataclass
class OcclusionResult:
    impacts: torch.Tensor  # (B,L,C), original forecast - occluded forecast
    target_at_input: torch.Tensor  # (B,)


@torch.no_grad()
def temporal_occlusion(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    *,
    horizon: int,
    target_channel: int = 0,
    target_step: int = -1,
    block_size: int = 10,
) -> OcclusionResult:
    """Thay từng block input của từng channel bằng baseline từ TRAIN.

    Impact âm: che block khiến dự báo tăng; impact dương: che khiến dự báo
    giảm. Các block sát nhau có thể trùng thông tin nên không cộng impact
    để suy ra độ đóng góp giống Integrated Gradients.
    """
    if model.training:
        raise ValueError("Occlusion yêu cầu model.eval()")
    if inputs.ndim != 3 or inputs.shape[0] == 0:
        raise ValueError("inputs phải có shape (B,L,C)")
    if block_size < 1:
        raise ValueError("block_size phải >= 1")
    try:
        base = torch.broadcast_to(baseline.to(device=inputs.device, dtype=inputs.dtype), inputs.shape)
    except RuntimeError as exc:
        raise ValueError("baseline không broadcast được đến inputs") from exc
    channels = inputs.shape[2]
    original = target_prediction(
        model(inputs), channels=channels, horizon=horizon,
        target_channel=target_channel, target_step=target_step,
    )
    impacts = torch.zeros_like(inputs)
    for start in range(0, inputs.shape[1], block_size):
        stop = min(start + block_size, inputs.shape[1])
        for channel in range(channels):
            changed = inputs.clone()
            changed[:, start:stop, channel] = base[:, start:stop, channel]
            changed_target = target_prediction(
                model(changed), channels=channels, horizon=horizon,
                target_channel=target_channel, target_step=target_step,
            )
            impacts[:, start:stop, channel] = (original - changed_target)[:, None]
    return OcclusionResult(impacts.detach().cpu(), original.detach().cpu())
