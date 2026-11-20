from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILES = {
    "Electricity": (
        "experiments/05_ablation/data/electricity_ablation.pt"
    ),
    "Financial": (
        "experiments/05_ablation/data/financial_ablation.pt"
    ),
    "CMIP6 Wind": (
        "experiments/05_ablation/data/cmip6_wind_ablation.pt"
    ),
    "CMIP6 Pressure": (
        "experiments/05_ablation/data/cmip6_pressure_candidate_2500.pt"
    ),
    "Lorenz": (
        "experiments/01_lorenz/results/lorenz_dataset.pt"
    ),
}

PATCH_LENGTHS = [70, 80, 90, 100, 110, 120, 130]
HORIZONS = [2, 4, 6, 8, 10, 12, 14, 16]


def load_raw_dataset(dataset_name: str):
    """Đọc chuỗi thời gian gốc từ dữ liệu đã chuẩn bị."""

    if dataset_name not in DATASET_FILES:
        raise ValueError(f"Dataset không hợp lệ: {dataset_name}")

    path = PROJECT_ROOT / DATASET_FILES[dataset_name]

    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy: {path}")

    # Đọc dữ liệu đã lưu trên CPU
    bundle = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    # Lorenz và các bộ ablation có cấu trúc lưu khác nhau
    if dataset_name == "Lorenz":
        raw = bundle["data"]
        split_index = int(len(raw) * 0.8)
    else:
        raw = bundle["raw"]
        split_index = int(bundle["split_index"])

    # Chuyển dữ liệu về mảng NumPy
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().numpy()

    raw = np.asarray(raw, dtype=np.float32)

    # Kiểm tra kích thước và giá trị dữ liệu
    if raw.ndim != 2:
        raise ValueError("Dữ liệu phải có dạng (Time, Channels)")

    if not np.isfinite(raw).all():
        raise ValueError("Dataset chứa NaN hoặc Inf")

    if not 0 < split_index < len(raw):
        raise ValueError("Train/Test split không hợp lệ")

    return raw, split_index, bundle


def split_and_scale(
    raw: np.ndarray,
    split_index: int,
    val_ratio: float = 0.2
):
    """Chia dữ liệu theo thời gian và chuẩn hóa chỉ từ train."""

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio phải nằm trong (0, 1)")

    # Tách validation từ phần train của ablation cũ
    train_end = int(split_index * (1 - val_ratio))

    train_raw = raw[:train_end]
    val_raw = raw[train_end:split_index]
    test_raw = raw[split_index:]

    if min(len(train_raw), len(val_raw), len(test_raw)) == 0:
        raise ValueError("Một trong các tập dữ liệu bị rỗng")

    # Chỉ sử dụng train để học thông số chuẩn hóa
    scaler = MinMaxScaler(feature_range=(0, 1))

    train = scaler.fit_transform(train_raw)
    val = scaler.transform(val_raw)
    test = scaler.transform(test_raw)

    # Chuyển về float32 để sử dụng với PyTorch
    train = train.astype(np.float32)
    val = val.astype(np.float32)
    test = test.astype(np.float32)

    return train, val, test, scaler


def create_windows(
    data: np.ndarray,
    seq_len: int,
    horizon: int
):
    """Tạo sliding windows theo định dạng nhãn của tác giả."""

    if seq_len < 1 or horizon < 1:
        raise ValueError("seq_len và horizon phải dương")

    if data.ndim != 2:
        raise ValueError("Dữ liệu phải có dạng (Time, Channels)")

    # Giữ nguyên cách đếm cửa sổ trong ablation giữa kỳ
    num_windows = len(data) - seq_len - horizon

    if num_windows <= 0:
        raise ValueError(
            "Không đủ dữ liệu để tạo sliding windows"
        )

    inputs = []
    targets = []

    for i in range(num_windows):

        # Lấy chuỗi quá khứ làm đầu vào
        x = data[i:i + seq_len]

        # Lấy horizon thời điểm tiếp theo làm nhãn
        y = data[
            i + seq_len:
            i + seq_len + horizon
        ]

        # Chuyển nhãn về thứ tự feature-major
        y = y.T.reshape(-1)

        inputs.append(x)
        targets.append(y)

    # Đóng gói dữ liệu thành tensor
    x = torch.from_numpy(
        np.stack(inputs).astype(np.float32)
    )

    y = torch.from_numpy(
        np.stack(targets).astype(np.float32)
    )

    return x, y


def prepare_dataset(
    dataset_name: str,
    seq_len: int,
    horizon: int,
    val_ratio: float = 0.2
):
    """Chuẩn bị dữ liệu cho một cấu hình thí nghiệm."""

    if seq_len not in PATCH_LENGTHS:
        raise ValueError("Lookback không thuộc grid thí nghiệm")

    if horizon not in HORIZONS:
        raise ValueError("Horizon không thuộc grid thí nghiệm")

    # Đọc chuỗi dữ liệu đã lưu từ các thí nghiệm trước
    raw, split_index, bundle = load_raw_dataset(
        dataset_name
    )

    # Chia train, validation, test và chuẩn hóa
    train, val, test, scaler = split_and_scale(
        raw=raw,
        split_index=split_index,
        val_ratio=val_ratio
    )

    # Tạo các cửa sổ riêng để không vượt ranh giới tập
    x_train, y_train = create_windows(
        train, seq_len, horizon
    )

    x_val, y_val = create_windows(
        val, seq_len, horizon
    )

    x_test, y_test = create_windows(
        test, seq_len, horizon
    )

    # Kiểm tra số lượng biến và kích thước nhãn
    input_dim = raw.shape[1]

    for x, y in [
        (x_train, y_train),
        (x_val, y_val),
        (x_test, y_test)
    ]:
        if x.shape[0] != y.shape[0]:
            raise ValueError("Số mẫu đầu vào và nhãn không khớp")

        if x.shape[1:] != (seq_len, input_dim):
            raise ValueError("Kích thước input không hợp lệ")

        if y.shape[1] != horizon * input_dim:
            raise ValueError("Kích thước target không hợp lệ")

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "input_dim": input_dim,
        "seq_len": seq_len,
        "horizon": horizon,
        "dataset_name": dataset_name,
        "scaler": scaler
    }
