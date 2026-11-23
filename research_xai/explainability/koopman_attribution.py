from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KoopmanDiagnostics:
    singular_values: torch.Tensor  # (D,)
    spectral_radius: float
    centered_norm: torch.Tensor  # (B,H) for M2/M3/M4, else (B,2) for M1
    shift_gap: torch.Tensor | None  # (B,H): nan at t0, M1=None
    relative_shift_gap: torch.Tensor | None
    equilibrium: torch.Tensor | None
    states: torch.Tensor  # (B,H,D) or (B,2,D)


@torch.no_grad()
def koopman_diagnostics(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    horizon: int,
    variant: str,
) -> KoopmanDiagnostics | None:
    """Đánh giá dynamics trên test windows, tuyệt đối không cập nhật model.

    targets là feature-major (B,C*H). Ta chỉ dùng future thật để tạo
    latent quan sát, KHÔNG đưa future thật vào quá trình dự báo.
    """
    if variant == "M0":
        return None
    if variant not in {"M1", "M2", "M3", "M4"}:
        raise ValueError(f"Model không hợp lệ: {variant}")
    if model.training:
        raise ValueError("Dynamics yêu cầu model.eval()")
    if inputs.ndim != 3:
        raise ValueError("inputs phải là (B,L,C)")
    batch, lookback, channels = inputs.shape
    if targets.shape != (batch, channels * horizon):
        raise ValueError("targets không đúng shape feature-major")

    K, singular_values = model.koop.get_operator()
    spectral_radius = torch.linalg.eigvals(K.float()).abs().max().item()
    z_initial = model.backbone(inputs)
    equilibrium = model.koop.mu.detach() if variant == "M4" else None

    if variant == "M1":
        z_next = model.koop.step(z_initial, K)
        states = torch.stack((z_initial, z_next), dim=1)
        center = torch.zeros_like(z_initial)  # M1 Koopman thuần tuyến tính
        centered_norm = torch.linalg.vector_norm(states - center[:, None], dim=-1)
        return KoopmanDiagnostics(
            singular_values=singular_values.detach().cpu(),
            spectral_radius=spectral_radius,
            centered_norm=centered_norm.cpu(),
            shift_gap=None,
            relative_shift_gap=None,
            equilibrium=None,
            states=states.cpu(),
        )

    _, states = model(inputs, return_states=True)
    if states.shape[:2] != (batch, horizon):
        raise ValueError("Model trả về rollout state có shape không hợp lệ")
    center = equilibrium if equilibrium is not None else torch.zeros_like(z_initial[0])
    centered_norm = torch.linalg.vector_norm(states - center.view(1, 1, -1), dim=-1)

    # Tạo teacher-free target từ student backbone trên shifted real windows.
    true_future = targets.reshape(batch, channels, horizon).transpose(1, 2)
    entire = torch.cat((inputs, true_future), dim=1)
    shifted = torch.stack([entire[:, t:t + lookback] for t in range(horizon)], dim=1)
    shifted = shifted.reshape(batch * horizon, lookback, channels)
    reencoded = model.backbone(shifted).reshape(batch, horizon, -1)
    distance = torch.linalg.vector_norm(states - reencoded, dim=-1)
    relative = distance / torch.linalg.vector_norm(reencoded, dim=-1).clamp_min(1e-8)
    # t0 bằng đầu vào gốc; không đánh đồng với dự đoán 1 bước
    distance[:, 0] = float("nan")
    relative[:, 0] = float("nan")

    return KoopmanDiagnostics(
        singular_values=singular_values.detach().cpu(),
        spectral_radius=spectral_radius,
        centered_norm=centered_norm.detach().cpu(),
        shift_gap=distance.detach().cpu(),
        relative_shift_gap=relative.detach().cpu(),
        equilibrium=equilibrium.cpu() if equilibrium is not None else None,
        states=states.detach().cpu(),
    )
