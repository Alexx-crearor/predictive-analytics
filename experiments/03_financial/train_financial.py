
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

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "financial_train_test.pt"

bundle = torch.load(
    DATA_PATH,
    map_location="cpu",
    weights_only=False
)

train_data = bundle["train"]
test_data = bundle["test"]

INPUT_DIM = 2

PATCH_LENGTHS = [40, 50, 60, 70, 80, 90, 100]
HORIZONS = [2, 4, 6, 8, 10, 12, 14, 16]
MODELS = ["LSTM", "PatchTST", "Autoformer", "Informer"]

D_MODEL = 48
EPOCHS = 4000
LEARNING_RATE = 3e-4
LYAP_WEIGHT = 0.1

RESULTS_DIR = BASE_DIR / "results" / "financial_grid_search_e4000"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = RESULTS_DIR / "metrics.csv"

METRIC_COLUMNS = [
    "Model", "PatchLen", "Horizon",
    "d_model", "Set", "Epochs", "MSE", "MAE"
]

CONFIGURATIONS = list(product(
    PATCH_LENGTHS, HORIZONS, MODELS
))


def orthogonalize(W: torch.Tensor) -> torch.Tensor:
    Q, R = torch.linalg.qr(W)
    return Q


class StrictStableKoopmanOperator(nn.Module):

    def __init__(self, latent_dim: int, rho_max: float = 0.99):
        super().__init__()

        self.U_raw = nn.Parameter(
            torch.randn(latent_dim, latent_dim)
        )

        self.V_raw = nn.Parameter(
            torch.randn(latent_dim, latent_dim)
        )

        self.S_raw = nn.Parameter(
            torch.randn(latent_dim)
        )

        self.rho_max = rho_max

    def forward(self, z: torch.Tensor):

        # Orthogonal matrices
        U = orthogonalize(self.U_raw)
        V = orthogonalize(self.V_raw)

        # Constrained singular values
        Sigma = torch.sigmoid(self.S_raw) * self.rho_max

        # Construct Koopman operator
        K = U @ torch.diag(Sigma) @ V.T

        # Transform latent representations
        z_next = z @ K.T

        return z_next, K, Sigma


class SinCosPosEnc(nn.Module):

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()

        # Vị trí từ 0 đến max_len - 1
        pos = torch.arange(max_len).float().unsqueeze(1)

        # Hệ số tần số cho từng cặp chiều
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-np.log(10000.0) / d_model)
        )

        # Khởi tạo positional encoding
        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        # Lưu vào model nhưng không coi là trainable parameter
        self.register_buffer("pe", pe)

    def forward(self, x):
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

        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride
        )

    def forward(self, x):

        # (Batch, Time, Channels) -> (Batch, Channels, Time)
        x = x.permute(0, 2, 1)

        # Convolution tạo patch embeddings
        x = self.conv(x)

        # (Batch, d_model, N_patches) -> (Batch, N_patches, d_model)
        x = x.permute(0, 2, 1)

        return x


class PatchTST_Backbone(nn.Module):

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        patch_len: int,
        d_model: int = 48,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_ff: int = 96
    ):
        super().__init__()

        # 1. Chia chuỗi thành các patch và embedding
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

        # (B, seq_len, input_dim) -> (B, n_patches, d_model)
        x = self.patch(x)

        # Mở rộng CLS token cho toàn bộ batch
        cls = self.cls.expand(x.size(0), -1, -1)

        # Ghép CLS token vào đầu chuỗi
        x = torch.cat([cls, x], dim=1)

        # Positional Encoding + Transformer Encoder
        x = self.encoder(self.pos(x))

        # Lấy biểu diễn tại CLS token
        z = x[:, 0]

        return z


class Koopformer_PatchTST(nn.Module):

    def __init__(
        self,
        input_dim,
        seq_len,
        horizon,
        patch_len,
        d_model=48
    ):
        super().__init__()

        # PatchTST backbone
        self.backbone = PatchTST_Backbone(
            input_dim=input_dim,
            seq_len=seq_len,
            patch_len=patch_len,
            d_model=d_model
        )

        # Koopman operator
        self.koop = StrictStableKoopmanOperator(
            latent_dim=d_model
        )

        # Forecasting head
        self.fc = nn.Linear(
            d_model,
            horizon * input_dim
        )

    def forward(self, x, return_latents=False):

        # Encode input into latent space
        z = self.backbone(x)

        # Apply Koopman operator
        z_next, K, Sigma = self.koop(z)

        # Forecast future values
        pred = self.fc(z_next)

        if return_latents:
            return pred, z, z_next

        return pred


class SeriesDecomp(nn.Module):

    def __init__(self, k: int = 3):
        super().__init__()

        self.avg = nn.AvgPool1d(
            kernel_size=k,
            stride=1,
            padding=k // 2
        )

    def forward(self, x):

        # (B, T, C) -> (B, C, T)
        x_transposed = x.transpose(1, 2)

        # Moving Average -> Trend
        trend = self.avg(x_transposed)

        # (B, C, T) -> (B, T, C)
        trend = trend.transpose(1, 2)

        # Với kernel chẵn, AvgPool có thể sinh thêm 1 time step
        if trend.size(1) != x.size(1):
            trend = trend[:, :x.size(1)]

        # Seasonal = Original - Trend
        seasonal = x - trend

        return trend, seasonal


class SimpleAutoformer(nn.Module):

    def __init__(
        self,
        input_len,
        horizon,
        input_dim,
        patch_len,
        d_model=48,
        num_heads=4,
        dim_ff=64,
        num_layers=3
    ):
        super().__init__()

        # 1. Series Decomposition
        self.dec = SeriesDecomp(k=patch_len)

        # 2. Seasonal branch
        self.embed = nn.Linear(input_dim, d_model)

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

        # 3. Forecasting heads for both components
        self.fc_seas = nn.Linear(d_model, horizon)

        self.fc_trend = nn.Linear(
            input_len * input_dim,
            horizon
        )

    def forward(self, x):

        # Separate trend and seasonal components
        trend, seas = self.dec(x)

        # Seasonal branch: Transformer Encoder
        seas = self.embed(seas)
        seas = self.pos(seas)
        seas = self.encoder(seas)

        # Aggregate seasonal representations
        seas_o = self.fc_seas(seas.mean(dim=1))

        # Trend branch: Flatten + Linear
        trend_flat = trend.reshape(trend.size(0), -1)
        trend_o = self.fc_trend(trend_flat)

        # Combine both branches
        return seas_o + trend_o


class Koopformer_Autoformer(nn.Module):

    def __init__(
        self,
        input_dim,
        seq_len,
        horizon,
        patch_len,
        d_model=48
    ):
        super().__init__()

        # Autoformer backbone
        self.backbone = SimpleAutoformer(
            input_len=seq_len,
            horizon=horizon,
            input_dim=input_dim,
            patch_len=patch_len,
            d_model=d_model
        )

        # Koopman operates in H-dimensional latent space
        self.koop = StrictStableKoopmanOperator(
            latent_dim=horizon
        )

        # Forecasting head
        self.fc = nn.Linear(
            horizon,
            horizon * input_dim
        )

    def forward(self, x, return_latents=False):

        z = self.backbone(x)

        z_next, K, Sigma = self.koop(z)

        pred = self.fc(z_next)

        if return_latents:
            return pred, z, z_next

        return pred


class InformerSparse(nn.Module):

    def __init__(
        self,
        input_dim,
        d_model=48,
        num_heads=4,
        dim_ff=96,
        num_layers=3,
        seq_len=120,
        patch_len=1
    ):
        super().__init__()

        # Có sử dụng Patch Embedding hay không?
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

        # Positional Encoding
        self.pos = SinCosPosEnc(
            d_model=d_model,
            max_len=n_tokens
        )

        # Transformer Encoder
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

        # 4. Mean Pooling across tokens
        z = x.mean(dim=1)

        return z


class Koopformer_Informer(nn.Module):

    def __init__(
        self,
        input_dim,
        seq_len,
        horizon,
        patch_len,
        d_model=48
    ):
        super().__init__()

        # Informer Backbone
        self.backbone = InformerSparse(
            input_dim=input_dim,
            d_model=d_model,
            seq_len=seq_len,
            patch_len=patch_len
        )

        # Koopman Operator
        self.koop = StrictStableKoopmanOperator(
            latent_dim=d_model
        )

        # Forecasting Head
        self.fc = nn.Linear(
            d_model,
            horizon * input_dim
        )

    def forward(self, x, return_latents=False):

        # Extract latent representation
        z = self.backbone(x)

        # Apply Koopman transformation
        z_next, K, Sigma = self.koop(z)

        # Forecast BTC and ETH
        pred = self.fc(z_next)

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

        # 1. LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # 2. Forecasting head
        self.fc = nn.Linear(
            hidden_dim,
            horizon * input_dim
        )

    def forward(self, x, return_latents=False):

        # h_n: (num_layers, batch_size, hidden_dim)
        _, (h_n, c_n) = self.lstm(x)

        # Final hidden state of the last LSTM layer
        z = h_n[-1]

        # Forecast future values
        pred = self.fc(z)

        if return_latents:
            return pred, z, z

        return pred


def create_windows(data, seq_len, horizon):
    X, y = [], []

    for i in range(len(data) - seq_len - horizon):

        # p historical timestamps
        x_window = data[i:i + seq_len]

        # H future timestamps for each channel
        y_window = data[
            i + seq_len:i + seq_len + horizon
        ].T.reshape(-1)

        X.append(x_window)
        y.append(y_window)

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
    # 1. Prediction loss
    mse_loss = nn.functional.mse_loss(pred, target)

    # 2. Lyapunov regularization
    if lyap_weight > 0:

        # P = Identity matrix
        P = torch.eye(
            z.shape[1],
            device=z.device,
            dtype=z.dtype
        )

        # V(z) = z^T P z
        V_z = torch.einsum("bi,ij,bj->b", z, P, z)

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
        lyap_loss = torch.zeros_like(mse_loss)

    # 3. Total loss
    total_loss = mse_loss + lyap_weight * lyap_loss

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
    model = model.to(DEVICE)

    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    # Lưu lịch sử huấn luyện
    loss_history = []
    mse_history = []
    lyap_history = []
    eig_history = []

    log_interval = max(1, epochs // 10)

    for epoch in range(epochs):

        model.train()

        # 1. Reset gradients
        optimizer.zero_grad()

        # 2. Forward pass
        pred, z, z_next = model(
            X_train,
            return_latents=True
        )

        # 3. Compute loss
        total_loss, mse_loss, lyap_loss = compute_loss(
            pred=pred,
            target=y_train,
            z=z,
            z_next=z_next,
            lyap_weight=lyap_weight
        )

        # 4. Backpropagation
        total_loss.backward()

        # 5. Update parameters
        optimizer.step()

        # 6. Save training history
        loss_history.append(total_loss.item())
        mse_history.append(mse_loss.item())
        lyap_history.append(lyap_loss.item())

        # Theo dõi đại lượng giống script tác giả:
        # max(sigmoid(S_raw)), không phải eigenvalue thực của K
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

    return {
        "model": model,
        "loss": np.array(loss_history),
        "mse": np.array(mse_history),
        "lyap": np.array(lyap_history),
        "eig": np.array(eig_history)
    }


def evaluate_model(model, X, y, batch_size=256):
    model.eval()

    device = next(model.parameters()).device

    all_predictions = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_batch = X[i:i + batch_size].to(device)

            pred = model(x_batch)

            all_predictions.append(pred.cpu())

    predictions = torch.cat(all_predictions, dim=0)

    # Calculate evaluation metrics
    mse = torch.mean((predictions - y) ** 2).item()
    mae = torch.mean(torch.abs(predictions - y)).item()

    return {
        "MSE": mse,
        "MAE": mae,
        "predictions": predictions
    }

def create_model(model_name, patch_len, horizon, d_model=48):

    if model_name == "LSTM":
        model = SimpleLSTMForecaster(
            input_dim=INPUT_DIM,
            hidden_dim=d_model,
            num_layers=2,
            horizon=horizon
        )

    elif model_name == "PatchTST":
        model = Koopformer_PatchTST(
            input_dim=INPUT_DIM,
            seq_len=patch_len,
            horizon=horizon,
            patch_len=patch_len,
            d_model=d_model
        )

    elif model_name == "Autoformer":
        model = Koopformer_Autoformer(
            input_dim=INPUT_DIM,
            seq_len=patch_len,
            horizon=horizon,
            patch_len=patch_len,
            d_model=d_model
        )

    elif model_name == "Informer":
        model = Koopformer_Informer(
            input_dim=INPUT_DIM,
            seq_len=patch_len,
            horizon=horizon,
            patch_len=patch_len,
            d_model=d_model
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(DEVICE)


def run_one_configuration(model_name, patch_len, horizon):

    # 1. Checkpoint directory
    results_dir = (
        DATA_PATH.parent.parent
        / "results"
        / "financial_grid_search_e4000"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = results_dir / (
        f"{model_name.lower()}_p{patch_len}"
        f"_h{horizon}_d{D_MODEL}.pt"
    )

    # 2. Skip configurations already completed
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

        print(f"SKIP: {checkpoint_path.name}")

        return checkpoint_path

    print(
        f"\nSTART: {model_name} | "
        f"P={patch_len}, H={horizon}, "
        f"d_model={D_MODEL}"
    )

    # 3. Create sliding windows
    X_train, y_train = create_windows(
        train_data, patch_len, horizon
    )

    X_test, y_test = create_windows(
        test_data, patch_len, horizon
    )

    # 4. Initialize model
    model = create_model(
        model_name=model_name,
        patch_len=patch_len,
        horizon=horizon,
        d_model=D_MODEL
    )

    # 5. Select Lyapunov weight
    lyap_weight = (
        0.0 if model_name == "LSTM"
        else LYAP_WEIGHT
    )

    # 6. Train the model
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

    # 7. Evaluate Train and Test
    train_metrics = evaluate_model(
        trained_model, X_train, y_train
    )

    test_metrics = evaluate_model(
        trained_model, X_test, y_test
    )

    # 8. Build checkpoint
    checkpoint = {
        "model": model_name,

        "model_state_dict": {
            k: v.detach().cpu().clone()
            for k, v in trained_model.state_dict().items()
        },

        "pred_train": train_metrics["predictions"],
        "truth_train": y_train.cpu(),

        "pred_test": test_metrics["predictions"],
        "truth_test": y_test.cpu(),

        "loss": torch.tensor(result["loss"]),
        "eig": torch.tensor(result["eig"]),
        "mse_history": torch.tensor(result["mse"]),
        "lyap_history": torch.tensor(result["lyap"]),

        "train_mse": train_metrics["MSE"],
        "train_mae": train_metrics["MAE"],

        "test_mse": test_metrics["MSE"],
        "test_mae": test_metrics["MAE"],

        "patch_len": patch_len,
        "horizon": horizon,
        "d_model": D_MODEL,
        "input_dim": INPUT_DIM,

        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lyap_weight": lyap_weight,

        "channels": bundle["channels"],
        "data_hash": bundle["data_hash"],
    }

    # 9. Atomic checkpoint save
    temp_path = checkpoint_path.with_suffix(".pt.tmp")

    torch.save(checkpoint, temp_path)
    temp_path.replace(checkpoint_path)

    print(
        f"DONE: {model_name} | "
        f"P={patch_len}, H={horizon}"
    )

    print(
        f"Train MSE: {train_metrics['MSE']:.6f} | "
        f"Test MSE: {test_metrics['MSE']:.6f}"
    )

    print(f"Saved: {checkpoint_path}")

    # 10. Release GPU memory after completed run
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
    """
    Rebuild metrics.csv from completed checkpoints.
    Checkpoints are the source of truth.
    """

    rows = []
    completed = 0

    for patch_len, horizon, model_name in CONFIGURATIONS:

        path = checkpoint_file(
            model_name, patch_len, horizon
        )

        if not path.exists():
            continue

        ckpt = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        # Verify checkpoint identity
        assert ckpt["data_hash"] == bundle["data_hash"]
        assert ckpt["epochs"] == EPOCHS
        assert ckpt["model"] == model_name
        assert ckpt["patch_len"] == patch_len
        assert ckpt["horizon"] == horizon
        assert ckpt["d_model"] == D_MODEL

        completed += 1

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

    metrics_df = pd.DataFrame(
        rows, columns=METRIC_COLUMNS
    )

    # Atomic CSV save
    temp_path = METRICS_PATH.with_suffix(".csv.tmp")

    metrics_df.to_csv(temp_path, index=False)
    os.replace(temp_path, METRICS_PATH)

    return completed, metrics_df


def run_grid_search(max_new_runs=None):

    completed, _ = collect_metrics()

    print(f"Total configurations: {len(CONFIGURATIONS)}")
    print(f"Already completed: {completed}")
    print(f"Epochs per configuration: {EPOCHS}")
    print(f"Device: {DEVICE}")

    new_runs = 0

    for index, (patch_len, horizon, model_name) in enumerate(
        CONFIGURATIONS, start=1
    ):

        path = checkpoint_file(
            model_name, patch_len, horizon
        )

        if path.exists():
            print(f"SKIP [{index}/224] {path.name}")
            continue

        if max_new_runs is not None and new_runs >= max_new_runs:
            break

        print(
            f"\n[{index}/224] "
            f"{model_name} | P={patch_len}, H={horizon}"
        )

        # Train and save checkpoint
        run_one_configuration(
            model_name=model_name,
            patch_len=patch_len,
            horizon=horizon
        )

        new_runs += 1

        # Update CSV immediately after each completed run
        completed, metrics_df = collect_metrics()

        print(
            f"PROGRESS: {completed}/224 | "
            f"Metrics rows: {len(metrics_df)}"
        )

    completed, metrics_df = collect_metrics()

    print("\nGRID SEARCH STATUS")
    print("Completed:", completed, "/224")
    print("New runs:", new_runs)
    print("Metrics rows:", len(metrics_df))
    print("Results directory:", RESULTS_DIR)

    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum new configurations to train"
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be >= 1")

    print("Device:", DEVICE)
    print("GPU:", torch.cuda.get_device_name(0)
          if torch.cuda.is_available() else "CPU")
    print("Total configurations:", len(CONFIGURATIONS))
    print("Epochs:", EPOCHS)
    print("Dataset hash:", bundle["data_hash"])

    run_grid_search(max_new_runs=args.limit)
