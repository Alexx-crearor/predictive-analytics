
import tempfile

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

# Tạo dữ liệu nhỏ để kiểm tra logic huấn luyện
x_train = torch.randn(8, 20, 3)
y_train = torch.randn(8, 12)
x_val = torch.randn(4, 20, 3)
y_val = torch.randn(4, 12)

config = TrainConfig(
    epochs=10,
    batch_size=4,
    val_every=1,
    checkpoint_every=1,
    early_stopping_patience=2,
    early_stopping_min_delta=1e6
)

# Dùng thư mục tạm để không ảnh hưởng checkpoint chính
with tempfile.TemporaryDirectory() as run_dir:
    result = train_model(
        model=model,
        variant="M4",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        run_dir=run_dir,
        config=config,
        resume=False
    )

    print("Stopped epoch:", result["last_epoch"])
    print("Best epoch:", result["best_epoch"])
    print("Early stopped:", result["early_stopped"])

    assert result["early_stopped"] is True
    assert result["last_epoch"] == 3

print("EARLY STOPPING: OK")
