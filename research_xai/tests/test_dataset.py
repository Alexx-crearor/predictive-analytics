
import torch

from research_xai.data.dataset import (
    prepare_dataset,
    PATCH_LENGTHS,
    HORIZONS
)


def test_dataset(dataset_name):

    # Kiểm thử một cấu hình thuộc grid
    data = prepare_dataset(
        dataset_name=dataset_name,
        seq_len=120,
        horizon=8
    )

    print(f"\nDataset: {dataset_name}")

    for split in ["train", "val", "test"]:
        x = data[f"x_{split}"]
        y = data[f"y_{split}"]

        print(f"{split}: X={tuple(x.shape)}, Y={tuple(y.shape)}")

        assert x.ndim == 3
        assert y.ndim == 2
        assert torch.isfinite(x).all()
        assert torch.isfinite(y).all()

    assert data["y_train"].shape[1] == 8 * data["input_dim"]

    print("Kiểm tra dữ liệu: OK")


if __name__ == "__main__":
    for name in [
        "Electricity",
        "Financial",
        "CMIP6 Wind",
        "CMIP6 Pressure",
        "Lorenz"
    ]:
        test_dataset(name)

    assert len(PATCH_LENGTHS) * len(HORIZONS) == 56

    print("\nTất cả dataset: OK")
