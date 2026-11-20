
import torch

from research_xai.models.forecasters import TemporalKoopman
from research_xai.training.trainer import (
    TrainConfig,
    set_seed,
    train_model,
)


set_seed(7)

model = TemporalKoopman(
    input_dim=3,
    seq_len=20,
    horizon=4,
    patch_len=20,
    d_model=16,
    affine=True
)

# Tạo dữ liệu nhỏ để kiểm tra training
x_train = torch.randn(8, 20, 3)
y_train = torch.randn(8, 12)
x_val = torch.randn(4, 20, 3)
y_val = torch.randn(4, 12)

mu_before = model.koop.mu.detach().clone()

config = TrainConfig(
    epochs=2,
    batch_size=4,
    val_every=1,
    checkpoint_every=1
)

result = train_model(
    model=model,
    variant="M4",
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    run_dir="research_xai/results/smoke_test_m4",
    config=config,
    resume=False
)


# Chuyển mu sau huấn luyện về CPU để so sánh
mu_after = model.koop.mu.detach().cpu().clone()

# Tính độ thay đổi của điểm cân bằng
mu_change = (mu_after - mu_before).norm().item()

print("Best validation:", result["best_val_mse"])
print("Thay đổi mu:", mu_change)

assert mu_change > 0, "Điểm cân bằng mu chưa được cập nhật"

print("M4 training: OK")
