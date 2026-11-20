
import torch

from research_xai.models.forecasters import (
    PatchTSTDirect,
    KoopmanDirect,
    TemporalKoopman,
)


# Tạo dữ liệu kiểm thử
x = torch.randn(4, 120, 5)

models = {
    "M0": PatchTSTDirect(5, 120, 8, 120),
    "M1": KoopmanDirect(5, 120, 8, 120),
    "M2": TemporalKoopman(5, 120, 8, 120),
    "M3": TemporalKoopman(5, 120, 8, 120),
    "M4": TemporalKoopman(5, 120, 8, 120, affine=True),
}

for name, model in models.items():
    model.eval()

    with torch.no_grad():
        pred = model(x)

    assert pred.shape == (4, 40)
    print(f"{name}: {pred.shape}")

# Kiểm tra các trạng thái latent của mô hình cải tiến
model = models["M4"]

with torch.no_grad():
    pred, states = model(x, return_states=True)

assert states.shape == (4, 8, 96)

print("Predictions:", pred.shape)
print("Latent states:", states.shape)
print("Learnable mu:", model.koop.mu.shape)
