import argparse
import ast
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path("/root/deepkoopformer")

DATA = ROOT / "data/scenario4_wind_train_test.pt"
MODEL_NOTEBOOK = ROOT / "models/02_climate_model.ipynb"
OUT = ROOT / "results/scenario4_train_test_e4000"

OUT.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=1)
args = parser.parse_args()

# ============================================================
# 2. LOAD MODEL CLASSES FROM EXISTING NOTEBOOK
# ============================================================

nb = json.loads(MODEL_NOTEBOOK.read_text(encoding="utf-8"))

nodes = []

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue

    source = cell["source"]
    if isinstance(source, list):
        source = "".join(source)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        continue

    for node in tree.body:
        if isinstance(node, (
            ast.Import,
            ast.ImportFrom,
            ast.ClassDef,
            ast.FunctionDef,
            ast.AsyncFunctionDef
        )):
            nodes.append(node)

module = ast.Module(body=nodes, type_ignores=[])
ast.fix_missing_locations(module)

exec(
    compile(module, str(MODEL_NOTEBOOK), "exec"),
    globals()
)

# ============================================================
# 3. LOAD SCENARIO 4 DATASET
# ============================================================

bundle = torch.load(
    DATA,
    map_location="cpu",
    weights_only=True
)

train_data = bundle["train_scaled"].numpy()
test_data = bundle["test_scaled"].numpy()

seq_len = int(bundle["seq_len"])
patch_lengths = list(bundle["patch_lengths"])
horizons = list(bundle["horizons"])

d_model = int(bundle["d_model"])
epochs = int(bundle["epochs"])
lr = float(bundle["learning_rate"])
lyap_weight = float(bundle["lyap_weight"])
seed = int(bundle["seed"])

input_dim = train_data.shape[1]

device = torch.device("cuda:0")

assert torch.cuda.is_available()
assert epochs == 4000
assert seq_len == 120
assert d_model == 48
assert input_dim == 5

model_names = ["LSTM", "PatchTST", "Autoformer", "Informer"]

configs = list(itertools.product(
    patch_lengths, horizons, model_names
))

assert len(configs) == 224

data_hash = hashlib.sha256(
    train_data.tobytes() + test_data.tobytes()
).hexdigest()

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("Total configurations:", len(configs), flush=True)
print("Epochs:", epochs, flush=True)

# ============================================================
# 4. SLIDING WINDOWS
# ============================================================

def build_windows(data, horizon):
    X, Y = [], []

    for i in range(len(data) - seq_len - horizon):
        X.append(data[i:i + seq_len])

        Y.append(
            data[i + seq_len:i + seq_len + horizon]
            .T.reshape(-1)
        )

    return (
        torch.tensor(np.stack(X), dtype=torch.float32),
        torch.tensor(np.stack(Y), dtype=torch.float32)
    )

# ============================================================
# 5. MODEL FACTORY
# ============================================================

def create_model(name, patch_len, horizon):

    if name == "LSTM":
        return SimpleLSTMForecaster(
            input_dim=input_dim,
            hidden_dim=48,
            num_layers=2,
            horizon=horizon
        )

    classes = {
        "PatchTST": Koopformer_PatchTST,
        "Autoformer": Koopformer_Autoformer,
        "Informer": Koopformer_Informer
    }

    return classes[name](
        input_dim,
        seq_len,
        horizon,
        patch_len=patch_len,
        d_model=d_model
    )

# ============================================================
# 6. CHECKPOINT AND METRICS
# ============================================================

def checkpoint_path(p, h, name):
    return OUT / f"{name.lower()}_p{p}_h{h}_d{d_model}.pt"

def calculate_metrics(pred, truth):
    error = pred - truth
    return (
        float(np.mean(error ** 2)),
        float(np.mean(np.abs(error)))
    )

def update_metrics():
    rows = []

    for p, h, name in configs:
        path = checkpoint_path(p, h, name)

        if not path.exists():
            continue

        ckpt = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

        assert ckpt["data_hash"] == data_hash
        assert ckpt["epochs"] == 4000
        assert ckpt["model"] == name
        assert ckpt["patch_len"] == p
        assert ckpt["horizon"] == h

        for split in ("train", "test"):
            rows.append({
                "Model": name,
                "PatchLen": p,
                "Horizon": h,
                "d_model": d_model,
                "Set": split.title(),
                "Epochs": epochs,
                "MSE": ckpt[f"{split}_mse"],
                "MAE": ckpt[f"{split}_mae"]
            })

    metrics_path = OUT / "metrics.csv"
    temp_path = OUT / "metrics.csv.tmp"

    pd.DataFrame(rows, columns=[
        "Model", "PatchLen", "Horizon",
        "d_model", "Set", "Epochs", "MSE", "MAE"
    ]).to_csv(temp_path, index=False)

    os.replace(temp_path, metrics_path)

    return len(rows) // 2

# ============================================================
# 7. TRAINING LOOP
# ============================================================

new_runs = 0

for run_idx, (p, h, name) in enumerate(configs, start=1):

    path = checkpoint_path(p, h, name)

    if path.exists():
        ckpt = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

        assert ckpt["data_hash"] == data_hash
        assert ckpt["epochs"] == epochs
        assert ckpt["model"] == name
        assert ckpt["patch_len"] == p
        assert ckpt["horizon"] == h
        assert ckpt["seq_len"] == seq_len
        assert ckpt["d_model"] == d_model

        if "rng_torch" in ckpt:
            torch.set_rng_state(ckpt["rng_torch"])

        if ckpt.get("rng_cuda"):
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])

        print(f"SKIP [{run_idx}/224] {path.name}", flush=True)
        continue

    if new_runs >= args.limit:
        break

    print(
        f"\nSTART [{run_idx}/224] "
        f"{name} | P={p}, H={h}",
        flush=True
    )

    start_time = time.monotonic()

    x_train, y_train = build_windows(train_data, h)
    x_test, y_test = build_windows(test_data, h)

    model = create_model(name, p, h).to(device)

    x_gpu = x_train.to(device)
    y_gpu = y_train.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    mse = nn.MSELoss()

    with torch.no_grad():
        _, z, _ = model(
            x_gpu[:1],
            return_latents=True
        )

    P = torch.eye(z.shape[1], device=device)

    weight = 0.0 if name == "LSTM" else lyap_weight

    losses = []
    eigs = []

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred, z, z_next = model(
            x_gpu,
            return_latents=True
        )

        pred_loss = mse(pred, y_gpu)

        if weight:
            zp = torch.einsum("bi,ij->bj", z, P)
            znp = torch.einsum("bi,ij->bj", z_next, P)

            lyap = torch.relu(
                (znp * z_next).sum(1)
                - (zp * z).sum(1)
            ).mean()
        else:
            lyap = torch.zeros((), device=device)

        loss = pred_loss + weight * lyap

        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

        koop = getattr(model, "koop", None)

        if isinstance(koop, StrictStableKoopmanOperator):
            eigs.append(
                float(torch.sigmoid(koop.S_raw).abs().max().item())
            )
        else:
            eigs.append(0.0)

        if ep % 400 == 0 or ep == epochs - 1:
            print(
                f"Epoch {ep + 1:4d}/{epochs} "
                f"MSE={pred_loss.item():.6f} "
                f"Lyap={(weight * lyap).item():.6f}",
                flush=True
            )

    # Evaluation: no parameter updates
    model.eval()

    with torch.no_grad():
        pred_train = model(x_gpu).cpu().numpy()
        pred_test = model(x_test.to(device)).cpu().numpy()

    truth_train = y_train.numpy()
    truth_test = y_test.numpy()

    train_mse, train_mae = calculate_metrics(
        pred_train, truth_train
    )

    test_mse, test_mae = calculate_metrics(
        pred_test, truth_test
    )

    assert np.isfinite([
        train_mse, train_mae, test_mse, test_mae
    ]).all()

    checkpoint = {
        "model": name,
        "model_state_dict": {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        },
        "pred_train": torch.from_numpy(pred_train.copy()),
        "truth_train": torch.from_numpy(truth_train.copy()),
        "pred_test": torch.from_numpy(pred_test.copy()),
        "truth_test": torch.from_numpy(truth_test.copy()),
        "loss": torch.tensor(losses),
        "eig": torch.tensor(eigs),
        "train_mse": train_mse,
        "train_mae": train_mae,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "patch_len": p,
        "horizon": h,
        "seq_len": seq_len,
        "d_model": d_model,
        "input_dim": input_dim,
        "epochs": epochs,
        "data_hash": data_hash,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all()
    }

    # Atomic checkpoint saving
    temp_path = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)

    completed = update_metrics()
    torch.cuda.synchronize()

    print(
        f"DONE [{run_idx}/224] "
        f"Train MSE={train_mse:.6f} "
        f"Test MSE={test_mse:.6f}",
        flush=True
    )

    print(
        f"Elapsed: {time.monotonic() - start_time:.1f}s "
        f"| Saved: {path.name} "
        f"| Completed: {completed}/224",
        flush=True
    )

    new_runs += 1

    del model, x_gpu, y_gpu, x_train, y_train
    del x_test, y_test, checkpoint

    torch.cuda.empty_cache()

print(f"\nFINISHED | New runs: {new_runs}", flush=True)

