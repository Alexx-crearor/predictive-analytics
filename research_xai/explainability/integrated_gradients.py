from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IGResult:
    attribution: torch.Tensor  # (B, L, C), signed
    completeness_error: torch.Tensor  # (B,), signed
    target_at_input: torch.Tensor  # (B,)
    target_at_baseline: torch.Tensor  # (B,)


def target_prediction(
    prediction: torch.Tensor, *, channels: int, horizon: int,
    target_channel: int = 0, target_step: int = -1,
) -> torch.Tensor:
    """Lấy y_hat[channel, lead] từ output xếp theo feature-major.

    Output đã dùng khi training: (B, C * H) == (B, C, H).reshape(B,-1).
    """
    if prediction.ndim != 2 or prediction.shape[1] != channels * horizon:
        raise ValueError("Shape dự báo phải là (B, channels * horizon)")
    if target_step < 0:
        target_step += horizon
    if not 0 <= target_channel < channels or not 0 <= target_step < horizon:
        raise ValueError("target_channel hoặc target_step vượt giới hạn")
    return prediction.reshape(prediction.shape[0], channels, horizon)[:, target_channel, target_step]


def integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    *,
    horizon: int,
    target_channel: int = 0,
    target_step: int = -1,
    steps: int = 32,
) -> IGResult:
    """Tích phân xấp xỉ bằng quy tắc hình thang, giữ nguyên từng sample.

    inputs (B,L,C); baseline broadcastable tới inputs (ví dụ mean train C).
    Không cập nhật tham số mô hình; model phải được đặt ở eval().
    """
    if inputs.ndim != 3 or inputs.shape[0] == 0:
        raise ValueError("inputs phải có shape (B,L,C), B>0")
    if steps < 2:
        raise ValueError("steps phải >=2")
    if model.training:
        raise ValueError("Integrated Gradients cần model.eval() để bỏ dropout")
    if not torch.is_floating_point(inputs):
        raise ValueError("inputs phải là float tensor")
    try:
        base = torch.broadcast_to(baseline.to(device=inputs.device, dtype=inputs.dtype), inputs.shape)
    except RuntimeError as exc:
        raise ValueError("baseline không broadcast được đến inputs") from exc
    channels = inputs.shape[-1]
    delta_input = inputs.detach() - base.detach()
    total_gradients = torch.zeros_like(inputs)

    # Không cần lưu graph từ mỗi bước, chỉ đạo hàm đối với input.
    for i in range(steps + 1):
        alpha = i / steps
        mixed = (base + alpha * delta_input).detach().requires_grad_(True)
        prediction = model(mixed)
        selected = target_prediction(
            prediction, channels=channels, horizon=horizon,
            target_channel=target_channel, target_step=target_step,
        )
        gradient = torch.autograd.grad(selected.sum(), mixed)[0]
        weight = 0.5 if i in (0, steps) else 1.0
        total_gradients.add_(gradient.detach(), alpha=weight / steps)

    attributions = delta_input * total_gradients
    with torch.no_grad():
        input_target = target_prediction(
            model(inputs), channels=channels, horizon=horizon,
            target_channel=target_channel, target_step=target_step,
        )
        baseline_target = target_prediction(
            model(base), channels=channels, horizon=horizon,
            target_channel=target_channel, target_step=target_step,
        )
    completeness_error = attributions.flatten(1).sum(1) - (input_target - baseline_target)
    return IGResult(
        attribution=attributions.detach().cpu(),
        completeness_error=completeness_error.detach().cpu(),
        target_at_input=input_target.detach().cpu(),
        target_at_baseline=baseline_target.detach().cpu(),
    )
