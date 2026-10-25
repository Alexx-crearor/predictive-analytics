
import argparse
import os
import random

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


# Reproducibility
SEED = 7

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Dataset
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR / "data" / "electricity_train_test.pt"
)

bundle = torch.load(
    DATA_PATH,
    map_location="cpu",
    weights_only=False
)

train_data = bundle["train"]
test_data = bundle["test"]

INPUT_DIM = train_data.shape[1]
CHANNELS = bundle["channels"]

assert train_data.shape == (2800, 6)
assert test_data.shape == (700, 6)
assert INPUT_DIM == 6


# Grid Search
PATCH_LENGTHS = [70, 80, 90, 100, 110, 120, 130]
HORIZONS = [2, 4, 6, 8, 10, 12, 14, 16]

MODELS = [
    "LSTM",
    "PatchTST",
    "Autoformer",
    "Informer"
]


# Hyperparameters
D_MODEL = 96
LSTM_HIDDEN_DIM = 48

EPOCHS = 4000
LEARNING_RATE = 3e-4
LYAP_WEIGHT = 0.1


# Results
RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "electricity_grid_search_e4000"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_PATH = RESULTS_DIR / "metrics.csv"

METRIC_COLUMNS = [
    "Model", "PatchLen", "Horizon",
    "d_model", "Set", "Epochs",
    "MSE", "MAE"
]

CONFIGURATIONS = list(
    product(
        PATCH_LENGTHS,
        HORIZONS,
        MODELS
    )
)

assert len(CONFIGURATIONS) == 224


def orthogonalize(W: torch.Tensor) -> torch.Tensor:
    Q, R = torch.linalg.qr(W)
    return Q


class StrictStableKoopmanOperator(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        rho_max: float = 0.99
    ):
        super().__init__()

        # Learnable matrices
        self.U_raw = nn.Parameter(
            torch.randn(latent_dim, latent_dim)
        )

        self.V_raw = nn.Parameter(
            torch.randn(latent_dim, latent_dim)
        )

        self.S_raw = nn.Parameter(
            torch.randn(latent_dim)
        )

        # Upper bound for singular values
        self.rho_max = rho_max

    def forward(self, z: torch.Tensor):

        # Step 1: Orthogonal matrices
        U = orthogonalize(self.U_raw)
        V = orthogonalize(self.V_raw)

        # Step 2: Constrained singular values
        Sigma = (
            torch.sigmoid(self.S_raw)
            * self.rho_max
        )

        # Step 3: Construct Koopman operator
        K = U @ torch.diag(Sigma) @ V.T

        # Step 4: Apply Koopman transformation
        z_next = z @ K.T

        return z_next, K, Sigma


class SinCosPosEnc(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_len: int = 10000
    ):
        super().__init__()

        # 1. Tạo vị trí: 0, 1, 2, ..., max_len - 1
        pos = torch.arange(
            max_len
        ).float().unsqueeze(1)

        # 2. Tính các hệ số tần số
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-np.log(10000.0) / d_model)
        )

        # 3. Khởi tạo ma trận Positional Encoding
        pe = torch.zeros(
            max_len,
            d_model
        )

        # 4. Mã hóa vị trí bằng sin và cos
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        # 5. Lưu tensor vào model
        self.register_buffer("pe", pe)

    def forward(self, x):

        # Cộng Positional Encoding vào Input Embeddings
        return x + self.pe[:x.size(1)]


class PatchEmbed1D(nn.Module):

    def __init__(
        self,
        in_ch: int,
        d_model: int,
        patch_len: int,
        stride: int
    ):
        super().__init__()

        # 1. Convolution để tạo patch embeddings
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride
        )

    def forward(self, x):

        # Input: (Batch, Time, Channels)
        # Conv1d yêu cầu: (Batch, Channels, Time)
        x = x.permute(0, 2, 1)

        # 2. Patch Embedding
        x = self.conv(x)

        # Output: (Batch, N_patches, d_model)
        x = x.permute(0, 2, 1)

        return x


class PatchTST_Backbone(nn.Module):

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        patch_len: int,
        d_model: int = 96,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_ff: int = 96
    ):
        super().__init__()

        # 1. Patch Embedding
        self.patch = PatchEmbed1D(
            in_ch=input_dim,
            d_model=d_model,
            patch_len=patch_len,
            stride=patch_len
        )

        # 2. Positional Encoding
        n_patches = int(np.ceil(seq_len / patch_len))

        self.pos = SinCosPosEnc(
            d_model=d_model,
            max_len=n_patches + 1
        )

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4. Learnable CLS Token
        self.cls = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

    def forward(self, x):

        # (B, seq_len, input_dim)
        # -> (B, n_patches, d_model)
        x = self.patch(x)

        # Thêm CLS Token vào đầu chuỗi
        cls = self.cls.expand(
            x.size(0), -1, -1
        )

        x = torch.cat(
            [cls, x],
            dim=1
        )

        # Positional Encoding + Transformer Encoder
        x = self.encoder(
            self.pos(x)
        )

        # Lấy representation của CLS Token
        return x[:, 0]


class Koopformer_PatchTST(nn.Module):

    def __init__(
        self,
        input_dim,
        seq_len,
        horizon,
        patch_len,
        d_model=96
    ):
        super().__init__()

        # 1. PatchTST Backbone
        self.backbone = PatchTST_Backbone(
            input_dim=input_dim,
            seq_len=seq_len,
            patch_len=patch_len,
            d_model=d_model
        )

        # 2. Strict Stable Koopman Operator
        self.koop = StrictStableKoopmanOperator(
            latent_dim=d_model
        )

        # 3. Forecasting Head
        self.fc = nn.Linear(
            d_model,
            horizon * input_dim
        )

    def forward(self, x, return_latents=False):

        # 1. Trích xuất latent representation
        z = self.backbone(x)

        # 2. Biến đổi latent bằng Koopman Operator
        z_next, K, Sigma = self.koop(z)

        # 3. Dự báo các bước tương lai
        pred = self.fc(z_next)

        # 4. Trả về latent nếu cần tính Lyapunov Loss
        if return_latents:
            return pred, z, z_next

        return pred


class SeriesDecomp(nn.Module):

    def __init__(self, k: int = 3):
        super().__init__()

        # Moving Average để trích xuất Trend
        self.avg = nn.AvgPool1d(
            kernel_size=k,
            stride=1,
            padding=k // 2
        )

    def forward(self, x):

        # 1. (B, T, C) -> (B, C, T)
        x_transposed = x.transpose(1, 2)

        # 2. Moving Average -> Trend
        trend = self.avg(x_transposed)

        # 3. (B, C, T) -> (B, T, C)
        trend = trend.transpose(1, 2)

        # 4. Với kernel chẵn, AvgPool có thể sinh thêm 1 timestep
        if trend.size(1) != x.size(1):
            trend = trend[:, :x.size(1)]

        # 5. Seasonal = Original - Trend
        seasonal = x - trend

        return trend, seasonal


class SimpleAutoformer(nn.Module):

    def __init__(
        self,
        input_len,
        horizon,
        input_dim,
        patch_len,
        d_model=96,
        num_heads=4,
        dim_ff=64,
        num_layers=3
    ):
        super().__init__()

        # 1. Series Decomposition
        self.dec = SeriesDecomp(
            k=patch_len
        )

        # 2. Seasonal Branch
        self.embed = nn.Linear(
            input_dim,
            d_model
        )

        self.pos = SinCosPosEnc(
            d_model=d_model,
            max_len=input_len
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 3. Forecasting Heads
        self.fc_seas = nn.Linear(
            d_model,
            horizon
        )

        self.fc_trend = nn.Linear(
            input_len * input_dim,
            horizon
        )

    def forward(self, x):

        # 1. Tách Trend và Seasonal
        trend, seas = self.dec(x)

        # 2. Seasonal Branch
        seas = self.embed(seas)
        seas = self.pos(seas)
        seas = self.encoder(seas)

        # Tổng hợp thông tin theo chiều thời gian
        seas_o = self.fc_seas(
            seas.mean(dim=1)
        )

        # 3. Trend Branch
        trend_o = self.fc_trend(
            trend.reshape(trend.size(0), -1)
        )

        # 4. Kết hợp hai thành phần
        return seas_o + trend_o


class Koopformer_Autoformer(nn.Module):

    def __init__(
        self,
        input_dim,
        seq_len,
        horizon,
        patch_len,
        d_model=96
    ):
        super().__init__()

        # 1. Autoformer Backbone
        self.backbone = SimpleAutoformer(
            input_len=seq_len,
            horizon=horizon,
            input_dim=input_dim,
            patch_len=patch_len,
            d_model=d_model
        )

        # 2. Strict Stable Koopman Operator
        # Latent dimension bằng Forecast Horizon
        self.koop = StrictStableKoopmanOperator(
            latent_dim=horizon
        )

        # 3. Forecasting Head
        self.fc = nn.Linear(
            horizon,
            horizon * input_dim
        )

    def forward(self, x, return_latents=False):

        # 1. Trích xuất latent representation
        z = self.backbone(x)

        # 2. Biến đổi bằng Koopman Operator
        z_next, K, Sigma = self.koop(z)

        # 3. Dự báo các nguồn phát điện
        pred = self.fc(z_next)

        if return_latents:
            return pred, z, z_next

        return pred


class InformerSparse(nn.Module):

    def __init__(
        self,
        input_dim,
        d_model=96,
        num_heads=4,
        dim_ff=96,
        num_layers=3,
        seq_len=120,
        patch_len=1
    ):
        super().__init__()

        # 1. Có sử dụng Patch Embedding hay không?
        self.use_patch = patch_len > 1

        if self.use_patch:

            self.patch = PatchEmbed1D(
                in_ch=input_dim,
                d_model=d_model,
                patch_len=patch_len,
                stride=patch_len
            )

            n_tokens = int(np.ceil(seq_len / patch_len))

        else:

            self.embed = nn.Linear(
                input_dim,
                d_model
            )

            n_tokens = seq_len

        # 2. Positional Encoding
        self.pos = SinCosPosEnc(
            d_model=d_model,
            max_len=n_tokens
        )

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):

        # 1. Input Embedding
        if self.use_patch:
            x = self.patch(x)
        else:
            x = self.embed(x)

        # 2. Positional Encoding
        x = self.pos(x)

        # 3. Transformer Encoder
        x = self.encoder(x)

        # 4. Mean Pooling
        return x.mean(dim=1)


class Koopformer_Informer(nn.Module):

    def __init__(
        self,
        input_dim,
        seq_len,
        horizon,
        patch_len,
        d_model=96
    ):
        super().__init__()

        # 1. Informer Backbone
        self.backbone = InformerSparse(
            input_dim=input_dim,
            d_model=d_model,
            seq_len=seq_len,
            patch_len=patch_len
        )

        # 2. Strict Stable Koopman Operator
        self.koop = StrictStableKoopmanOperator(
            latent_dim=d_model
        )

        # 3. Forecasting Head
        self.fc = nn.Linear(
            d_model,
            horizon * input_dim
        )

    def forward(self, x, return_latents=False):

        # 1. Extract latent representation
        z = self.backbone(x)

        # 2. Apply Koopman transformation
        z_next, K, Sigma = self.koop(z)

        # 3. Forecast future values
        pred = self.fc(z_next)

        # 4. Return latent representations if needed
        if return_latents:
            return pred, z, z_next

        return pred


class SimpleLSTMForecaster(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        horizon
    ):
        super().__init__()

        # 1. LSTM Layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # 2. Forecasting Head
        self.fc = nn.Linear(
            hidden_dim,
            horizon * input_dim
        )

    def forward(self, x, return_latents=False):

        # 1. LSTM Forward Pass
        _, (h_n, c_n) = self.lstm(x)

        # 2. Lấy hidden state của layer cuối
        z = h_n[-1]

        # 3. Forecasting
        pred = self.fc(z)

        # 4. Trả về latent nếu cần
        if return_latents:
            return pred, z, z

        return pred


def create_windows(data, seq_len, horizon):

    X, y = [], []

    for i in range(len(data) - seq_len - horizon):

        # Input: seq_len historical timestamps
        x_window = data[
            i : i + seq_len
        ]

        # Target: horizon future timestamps
        y_window = data[
            i + seq_len : i + seq_len + horizon
        ].T.reshape(-1)

        X.append(x_window)
        y.append(y_window)

    # Convert to PyTorch tensors
    X = torch.stack(X).float()
    y = torch.stack(y).float()

    return X, y


def compute_loss(
    pred,
    target,
    z,
    z_next,
    lyap_weight=0.1
):

    # 1. Prediction Loss
    mse_loss = nn.functional.mse_loss(
        pred,
        target
    )

    # 2. Lyapunov Regularization
    if lyap_weight > 0:

        # Identity matrix
        P = torch.eye(
            z.shape[1],
            device=z.device,
            dtype=z.dtype
        )

        # V(z) = z^T P z
        V_z = torch.einsum(
            "bi,ij,bj->b",
            z, P, z
        )

        # V(z_next) = z_next^T P z_next
        V_z_next = torch.einsum(
            "bi,ij,bj->b",
            z_next, P, z_next
        )

        # Penalize positive increase
        lyap_loss = torch.relu(
            V_z_next - V_z
        ).mean()

    else:

        lyap_loss = torch.zeros_like(
            mse_loss
        )

    # 3. Total Loss
    total_loss = (
        mse_loss
        + lyap_weight * lyap_loss
    )

    return total_loss, mse_loss, lyap_loss


def train_model(
    model,
    X_train,
    y_train,
    epochs=4000,
    lr=3e-4,
    lyap_weight=0.1,
    verbose=True
):

    # 1. Move model and data to device
    model = model.to(DEVICE)

    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)

    # 2. Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    # 3. Training histories
    loss_history = []
    mse_history = []
    lyap_history = []
    eig_history = []

    log_interval = max(1, epochs // 10)

    # 4. Training loop
    for epoch in range(epochs):

        model.train()

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        pred, z, z_next = model(
            X_train,
            return_latents=True
        )

        # Compute loss
        total_loss, mse_loss, lyap_loss = compute_loss(
            pred=pred,
            target=y_train,
            z=z,
            z_next=z_next,
            lyap_weight=lyap_weight
        )

        # Backpropagation
        total_loss.backward()

        # Update parameters
        optimizer.step()

        # 5. Save training histories
        loss_history.append(
            total_loss.item()
        )

        mse_history.append(
            mse_loss.item()
        )

        lyap_history.append(
            lyap_loss.item()
        )

        # 6. Track Koopman quantity
        if hasattr(model, "koop"):

            with torch.no_grad():

                eig_max = (
                    torch.sigmoid(model.koop.S_raw)
                    .abs()
                    .max()
                    .item()
                )

        else:
            eig_max = 0.0

        eig_history.append(eig_max)

        # 7. Print training progress
        if verbose and (
            epoch % log_interval == 0
            or epoch == epochs - 1
        ):

            print(
                f"Epoch {epoch+1:4d}/{epochs} | "
                f"MSE: {mse_loss.item():.6f} | "
                f"Lyap: {lyap_loss.item():.6f} | "
                f"Total: {total_loss.item():.6f}"
            )

    # 8. Return training results
    return {
        "model": model,
        "loss": np.array(loss_history),
        "mse": np.array(mse_history),
        "lyap": np.array(lyap_history),
        "eig": np.array(eig_history)
    }


def evaluate_model(
    model,
    X,
    y,
    batch_size=256
):

    # 1. Evaluation mode
    model.eval()

    device = next(
        model.parameters()
    ).device

    all_predictions = []

    # 2. Disable gradients
    with torch.no_grad():

        for i in range(
            0,
            len(X),
            batch_size
        ):

            x_batch = X[
                i : i + batch_size
            ].to(device)

            # Model predictions
            pred = model(x_batch)

            all_predictions.append(
                pred.cpu()
            )

    # 3. Concatenate predictions
    predictions = torch.cat(
        all_predictions,
        dim=0
    )

    # 4. Targets on CPU
    targets = y.cpu()

    # 5. Evaluation metrics
    mse = torch.mean(
        (predictions - targets) ** 2
    ).item()

    mae = torch.mean(
        torch.abs(predictions - targets)
    ).item()

    return {
        "MSE": mse,
        "MAE": mae,
        "predictions": predictions
    }


def create_model(
    model_name,
    patch_len,
    horizon
):

    if model_name == "LSTM":

        model = SimpleLSTMForecaster(
            input_dim=INPUT_DIM,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_layers=2,
            horizon=horizon
        )

    elif model_name == "PatchTST":

        model = Koopformer_PatchTST(
            input_dim=INPUT_DIM,
            seq_len=patch_len,
            horizon=horizon,
            patch_len=patch_len,
            d_model=D_MODEL
        )

    elif model_name == "Autoformer":

        model = Koopformer_Autoformer(
            input_dim=INPUT_DIM,
            seq_len=patch_len,
            horizon=horizon,
            patch_len=patch_len,
            d_model=D_MODEL
        )

    elif model_name == "Informer":

        model = Koopformer_Informer(
            input_dim=INPUT_DIM,
            seq_len=patch_len,
            horizon=horizon,
            patch_len=patch_len,
            d_model=D_MODEL
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model.to(DEVICE)

def run_one_configuration(
    model_name,
    patch_len,
    horizon
):

    # Step 1: Checkpoint Path
    checkpoint_path = RESULTS_DIR / (
        f"{model_name.lower()}_p{patch_len}"
        f"_h{horizon}_d{D_MODEL}.pt"
    )

    # Skip completed configurations
    if checkpoint_path.exists():

        saved = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

        assert saved["data_hash"] == bundle["data_hash"]
        assert saved["epochs"] == EPOCHS
        assert saved["model"] == model_name
        assert saved["patch_len"] == patch_len
        assert saved["horizon"] == horizon
        assert saved["d_model"] == D_MODEL
        assert saved["input_dim"] == INPUT_DIM

        print(f"SKIP: {checkpoint_path.name}")

        return checkpoint_path

    print(
        f"\nSTART: {model_name} | "
        f"P={patch_len}, H={horizon}"
    )

    # Step 2: Sliding Windows
    X_train, y_train = create_windows(
        train_data,
        patch_len,
        horizon
    )

    X_test, y_test = create_windows(
        test_data,
        patch_len,
        horizon
    )

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # Step 3: Initialize Model
    model = create_model(
        model_name=model_name,
        patch_len=patch_len,
        horizon=horizon
    )

    # Step 4: Lyapunov Weight
    lyap_weight = (
        0.0 if model_name == "LSTM"
        else LYAP_WEIGHT
    )

    # Step 5: Training
    result = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        lyap_weight=lyap_weight,
        verbose=True
    )

    trained_model = result["model"]

    # Step 6: Evaluation
    train_metrics = evaluate_model(
        trained_model,
        X_train,
        y_train
    )

    test_metrics = evaluate_model(
        trained_model,
        X_test,
        y_test
    )

    # Step 7: Build Checkpoint
    checkpoint = {

        # Model Information
        "model": model_name,

        "model_state_dict": {
            k: v.detach().cpu().clone()
            for k, v in trained_model.state_dict().items()
        },

        # Train Predictions
        "pred_train": train_metrics["predictions"],
        "truth_train": y_train.cpu(),

        # Test Predictions
        "pred_test": test_metrics["predictions"],
        "truth_test": y_test.cpu(),

        # Training Histories
        "loss": torch.tensor(result["loss"]),
        "eig": torch.tensor(result["eig"]),
        "mse_history": torch.tensor(result["mse"]),
        "lyap_history": torch.tensor(result["lyap"]),

        # Train Metrics
        "train_mse": train_metrics["MSE"],
        "train_mae": train_metrics["MAE"],

        # Test Metrics
        "test_mse": test_metrics["MSE"],
        "test_mae": test_metrics["MAE"],

        # Hyperparameters
        "patch_len": patch_len,
        "horizon": horizon,
        "d_model": D_MODEL,
        "lstm_hidden_dim": LSTM_HIDDEN_DIM,
        "input_dim": INPUT_DIM,

        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lyap_weight": lyap_weight,

        # Dataset Metadata
        "channels": CHANNELS,
        "data_hash": bundle["data_hash"],
    }

    # Step 8: Atomic Checkpoint Save
    temp_path = checkpoint_path.with_suffix(
        ".pt.tmp"
    )

    torch.save(checkpoint, temp_path)

    temp_path.replace(checkpoint_path)

    # Step 9: Print Results
    print(
        f"\nDONE: {model_name} | "
        f"P={patch_len}, H={horizon}"
    )

    print(
        f"Train MSE: {train_metrics['MSE']:.6f} | "
        f"Train MAE: {train_metrics['MAE']:.6f}"
    )

    print(
        f"Test MSE: {test_metrics['MSE']:.6f} | "
        f"Test MAE: {test_metrics['MAE']:.6f}"
    )

    print("Saved:", checkpoint_path)

    # Step 10: Release Memory
    del trained_model, model, result
    del X_train, y_train, X_test, y_test

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return checkpoint_path


def checkpoint_file(model_name, patch_len, horizon):

    return RESULTS_DIR / (
        f"{model_name.lower()}_p{patch_len}"
        f"_h{horizon}_d{D_MODEL}.pt"
    )


def collect_metrics():

    rows = []
    completed = 0

    for patch_len, horizon, model_name in CONFIGURATIONS:

        path = checkpoint_file(
            model_name,
            patch_len,
            horizon
        )

        # Bỏ qua cấu hình chưa huấn luyện
        if not path.exists():
            continue

        # Load checkpoint
        ckpt = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        # Kiểm tra checkpoint thuộc đúng thực nghiệm
        assert ckpt["data_hash"] == bundle["data_hash"]
        assert ckpt["epochs"] == EPOCHS
        assert ckpt["model"] == model_name
        assert ckpt["patch_len"] == patch_len
        assert ckpt["horizon"] == horizon
        assert ckpt["d_model"] == D_MODEL
        assert ckpt["input_dim"] == INPUT_DIM
        assert ckpt["lstm_hidden_dim"] == LSTM_HIDDEN_DIM

        completed += 1

        # Mỗi checkpoint có hai bộ metrics: Train và Test
        for dataset in ["Train", "Test"]:

            prefix = dataset.lower()

            rows.append({
                "Model": model_name,
                "PatchLen": patch_len,
                "Horizon": horizon,
                "d_model": D_MODEL,
                "Set": dataset,
                "Epochs": EPOCHS,
                "MSE": ckpt[f"{prefix}_mse"],
                "MAE": ckpt[f"{prefix}_mae"]
            })

    # Tạo DataFrame
    metrics_df = pd.DataFrame(
        rows,
        columns=METRIC_COLUMNS
    )

    # Lưu metrics.csv an toàn
    temp_path = METRICS_PATH.with_suffix(
        ".csv.tmp"
    )

    metrics_df.to_csv(
        temp_path,
        index=False
    )

    os.replace(
        temp_path,
        METRICS_PATH
    )

    return completed, metrics_df


def run_grid_search(max_new_runs=None):

    # Kiểm tra những cấu hình đã hoàn thành
    completed, _ = collect_metrics()

    print("=== ELECTRICITY GRID SEARCH ===")
    print("Total configurations:", len(CONFIGURATIONS))
    print("Already completed:", completed)
    print("Epochs per configuration:", EPOCHS)
    print("Device:", DEVICE)

    new_runs = 0

    # Duyệt toàn bộ 224 cấu hình
    for index, (patch_len, horizon, model_name) in enumerate(
        CONFIGURATIONS,
        start=1
    ):

        path = checkpoint_file(
            model_name,
            patch_len,
            horizon
        )

        # Skip nếu đã hoàn thành
        if path.exists():

            print(
                f"SKIP [{index}/224] {path.name}"
            )

            continue

        # Giới hạn số lượt huấn luyện mới
        if (
            max_new_runs is not None
            and new_runs >= max_new_runs
        ):
            break

        print(
            f"\n[{index}/224] "
            f"{model_name} | "
            f"P={patch_len}, H={horizon}"
        )

        # Huấn luyện một cấu hình
        run_one_configuration(
            model_name=model_name,
            patch_len=patch_len,
            horizon=horizon
        )

        new_runs += 1

        # Cập nhật metrics sau mỗi lượt huấn luyện
        completed, metrics_df = collect_metrics()

        print(
            f"PROGRESS: {completed}/224 | "
            f"Metrics rows: {len(metrics_df)}"
        )

    # Tổng hợp kết quả cuối
    completed, metrics_df = collect_metrics()

    print("\n=== GRID SEARCH STATUS ===")
    print("Completed:", completed, "/224")
    print("New runs:", new_runs)
    print("Metrics rows:", len(metrics_df))
    print("Results directory:", RESULTS_DIR)

    return metrics_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Electricity DeepKoopFormer Grid Search"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum number of new configurations to train"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without training"
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be >= 1")

    print("=== ELECTRICITY GRID SEARCH ===")

    print("Device:", DEVICE)

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    print("Train shape:", tuple(train_data.shape))
    print("Test shape:", tuple(test_data.shape))

    print("Total configurations:", len(CONFIGURATIONS))
    print("Epochs:", EPOCHS)

    print("Koopman d_model:", D_MODEL)
    print("LSTM hidden dimension:", LSTM_HIDDEN_DIM)

    print("Dataset hash:", bundle["data_hash"])

    if args.dry_run:

        completed, metrics_df = collect_metrics()

        print("\\nDRY RUN")
        print("Completed:", completed)
        print("Remaining:", len(CONFIGURATIONS) - completed)
        print("Metrics rows:", len(metrics_df))
        print("No training performed.")

    else:

        run_grid_search(
            max_new_runs=args.limit
        )
