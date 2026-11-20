
from copy import deepcopy

import torch

from research_xai.models.forecasters import TemporalKoopman
from research_xai.training.losses import (
    ForecastingLoss,
    TemporalAlignmentLoss,
)


torch.manual_seed(7)

model = TemporalKoopman(
    input_dim=3,
    seq_len=20,
    horizon=4,
    patch_len=20,
    d_model=16
)

# Tạo teacher từ backbone hiện tại
teacher = deepcopy(model.backbone)
teacher.eval()

for parameter in teacher.parameters():
    parameter.requires_grad_(False)

# Tạo dữ liệu để kiểm tra phép tính loss
inputs = torch.randn(2, 20, 3)
targets = torch.randn(2, 12)

predictions, states = model(inputs, return_states=True)

forecast_loss = ForecastingLoss()
alignment_loss = TemporalAlignmentLoss(max_steps=2)

loss_forecast = forecast_loss(predictions, targets)

loss_alignment = alignment_loss(
    states=states,
    past=inputs,
    future=targets,
    teacher_encoder=teacher,
    steps=[1, 3]
)

total_loss = loss_forecast + 0.1 * loss_alignment

# Kiểm tra gradient truyền được qua các bước Koopman
total_loss.backward()

assert torch.isfinite(total_loss)
assert model.koop.U_raw.grad is not None
assert model.koop.S_raw.grad is not None
assert all(p.grad is None for p in teacher.parameters())

print("Forecast Loss:", loss_forecast.item())
print("Alignment Loss:", loss_alignment.item())
print("Total Loss:", total_loss.item())
print("Gradient Koopman: OK")
print("Teacher không nhận gradient: OK")
