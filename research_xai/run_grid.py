import argparse
import csv
import gc
import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from research_xai.data.dataset import (
    prepare_dataset,
    PATCH_LENGTHS,
    HORIZONS,
)
from research_xai.models.forecasters import (
    PatchTSTDirect,
    KoopmanDirect,
    TemporalKoopman,
)
from research_xai.training.trainer import (
    TrainConfig,
    calculate_data_hash,
    set_seed,
    train_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "research_xai" / "results" / "reduced_grid_225_v1"
# SYNC_RUNTIME_225_V1: luu giao thuc 225; full grid dung --grid full.
REDUCED_PATCHES = (70, 100, 130)
REDUCED_HORIZONS = (2, 8, 16)

DATASETS = {
    "energy": "Electricity",
    "financial": "Financial",
    "wind": "CMIP6 Wind",
    "pressure": "CMIP6 Pressure",
    "lorenz": "Lorenz",
}

MODELS = ["M0", "M1", "M2", "M3", "M4"]


def create_model(variant, input_dim, seq_len, horizon):
    """Khởi tạo mô hình tương ứng với biến thể thí nghiệm."""

    kwargs = {
        "input_dim": input_dim,
        "seq_len": seq_len,
        "horizon": horizon,
        "patch_len": seq_len,
        "d_model": 96,
    }

    if variant == "M0":
        return PatchTSTDirect(**kwargs)

    if variant == "M1":
        return KoopmanDirect(**kwargs, p_max=0.99)

    if variant in {"M2", "M3", "M4"}:
        return TemporalKoopman(
            **kwargs,
            p_max=0.99,
            affine=(variant == "M4"),
        )

    raise ValueError(f"Mô hình không hợp lệ: {variant}")


def build_grid(args):
    """Tạo danh sách các cấu hình cần chạy."""

    datasets = (
        list(DATASETS)
        if args.dataset == "all"
        else [args.dataset]
    )

    models = (
        MODELS
        if args.model == "all"
        else [args.model]
    )

    patches = (
        (REDUCED_PATCHES if getattr(args, "grid", "reduced225") == "reduced225" else PATCH_LENGTHS)
        if args.patch is None
        else [args.patch]
    )

    horizons = (
        (REDUCED_HORIZONS if getattr(args, "grid", "reduced225") == "reduced225" else HORIZONS)
        if args.horizon is None
        else [args.horizon]
    )

    return [
        {
            "dataset_key": dataset,
            "dataset": DATASETS[dataset],
            "model": model,
            "patch": patch,
            "horizon": horizon,
            "seed": args.seed,
        }
        for dataset, model, patch, horizon in product(
            datasets, models, patches, horizons
        )
    ]


def get_run_dir(run, config):
    """Tạo đường dẫn riêng cho từng cấu hình."""

    return (
        RESULTS_ROOT
        / run["dataset_key"]
        / run["model"]
        / (
            f"p{run['patch']}_h{run['horizon']}"
            f"_e{config.epochs}_b{config.batch_size}"
            f"_s{run['seed']}"
        )
    )



def is_completed(run_dir, config, data_hash):
    """Chi bo qua run neu metrics va best checkpoint deu khop.

    Khong tu train de ghi de khi run da hoan thanh nhung bi lech config.
    """
    metrics_path = run_dir / "metrics.json"
    latest_path = run_dir / "checkpoint_latest.pt"
    best_path = run_dir / "checkpoint_best.pt"

    if metrics_path.is_file():
        if not best_path.is_file():
            raise RuntimeError(f"Run da co metrics nhung thieu best checkpoint: {run_dir}")
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        expected = (
            metrics.get("status") == "completed"
            and metrics.get("target_epochs") == config.epochs
            and 0 < metrics.get("completed_epochs", 0) <= config.epochs
            and metrics.get("config") == asdict(config)
            and metrics.get("data_hash") == data_hash
        )
        if not expected:
            raise RuntimeError(
                f"Run da co metrics nhung config/data hash khong khop; "
                f"KHONG train de ghi de: {run_dir}"
            )
        return True

    if best_path.is_file() and not latest_path.is_file():
        raise RuntimeError(f"Co best checkpoint nhung thieu metrics/latest: {run_dir}")
    return False


@torch.no_grad()
def evaluate_model(model, x_test, y_test, batch_size):
    """Tính MSE và MAE trên tập test."""

    device = next(model.parameters()).device
    model.eval()

    loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    squared_error = 0.0
    absolute_error = 0.0
    total_values = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs)
        errors = predictions - targets

        squared_error += errors.square().sum().item()
        absolute_error += errors.abs().sum().item()
        total_values += errors.numel()

    return {
        "test_mse": squared_error / total_values,
        "test_mae": absolute_error / total_values,
    }


def run_experiment(run, config):
    """Huấn luyện và đánh giá một cấu hình thí nghiệm."""

    dataset_name = run["dataset"]
    variant = run["model"]
    patch = run["patch"]
    horizon = run["horizon"]
    seed = run["seed"]

    run_dir = get_run_dir(run, config)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Chuẩn bị dữ liệu theo đúng lookback và horizon
    data = prepare_dataset(
        dataset_name=dataset_name,
        seq_len=patch,
        horizon=horizon,
    )

    data_hash = calculate_data_hash(
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
    )

    # Không chạy lại các cấu hình đã hoàn thành
    if is_completed(run_dir, config, data_hash):
        print(f"[SKIP] {dataset_name} | {variant} | P={patch} H={horizon}")
        return "skipped"

    # Cố định seed trước khi khởi tạo tham số mô hình
    set_seed(seed)

    model = create_model(
        variant=variant,
        input_dim=data["input_dim"],
        seq_len=patch,
        horizon=horizon,
    )

    print(
        f"[START] {dataset_name} | {variant} | "
        f"P={patch} H={horizon} | Seed={seed}"
    )

    # Huấn luyện và tự khôi phục checkpoint nếu có
    result = train_model(
        model=model,
        variant=variant,
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_val=data["x_val"],
        y_val=data["y_val"],
        run_dir=run_dir,
        config=config,
        resume=True,
    )

    if result.get("time_limited", False):
        print(f"[PAUSED] {dataset_name} | {variant} | P={patch} H={horizon}")
        return "paused"

    # Chỉ đánh giá test bằng checkpoint tốt nhất
    best_path = run_dir / "checkpoint_best.pt"

    checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])

    metrics = evaluate_model(
        model=model,
        x_test=data["x_test"],
        y_test=data["y_test"],
        batch_size=config.batch_size,
    )

    # Lưu kết quả cùng thông tin cấu hình

    # Ghi nhận kết quả cuối cùng kể cả khi dừng sớm
    summary = {
        **run,
        **metrics,
        "status": "completed",
        "target_epochs": config.epochs,
        "completed_epochs": result["last_epoch"],
        "epochs_saved": result["epochs_saved"],
        "early_stopped": result["early_stopped"],
        "best_epoch": result["best_epoch"],
        "best_val_mse": result["best_val_mse"],
        "config": asdict(config),
        "data_hash": data_hash
    }

    metrics_path = run_dir / "metrics.json"
    temp_path = run_dir / "metrics.json.tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    temp_path.replace(metrics_path)

    print(
        f"[DONE] {dataset_name} | {variant} | "
        f"MSE={metrics['test_mse']:.6f} | "
        f"MAE={metrics['test_mae']:.6f}"
    )

    # Giải phóng bộ nhớ trước khi chuyển sang run tiếp theo
    del model, data, checkpoint
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return "completed"


def save_plan(grid, config):
    """Xuất danh sách thí nghiệm để kiểm tra trước khi chạy."""

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    path = RESULTS_ROOT / "grid_plan.csv"

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset_key", "dataset", "model",
                "patch", "horizon", "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(grid)

    print(f"Đã lưu danh sách: {path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS],
        default="all",
    )
    parser.add_argument(
        "--model",
        choices=["all", *MODELS],
        default="all",
    )
    parser.add_argument("--patch", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--grid", choices=("reduced225", "full"), default="reduced225")
    parser.add_argument("--results-root", type=str, default=None)

    # Chỉ huấn luyện khi người dùng chủ động bật
    parser.add_argument("--execute", action="store_true")

    # Mặc định chỉ chạy một cấu hình để tránh chạy nhầm
    parser.add_argument("--max-runs", type=int, default=1)

    return parser.parse_args()


def main():
    global RESULTS_ROOT
    args = parse_args()
    if args.grid == "full":
        RESULTS_ROOT = PROJECT_ROOT / "research_xai" / "results" / "grid_v2_earlystop"
    if args.results_root:
        RESULTS_ROOT = Path(args.results_root).resolve()

    if args.patch is not None and args.patch not in PATCH_LENGTHS:
        raise ValueError("Lookback không thuộc grid")

    if args.horizon is not None and args.horizon not in HORIZONS:
        raise ValueError("Horizon không thuộc grid")

    if args.grid == "reduced225":
        if args.patch is not None and args.patch not in REDUCED_PATCHES:
            raise ValueError("Lookback khong thuoc 225 runs. Dung --grid full neu muon full grid.")
        if args.horizon is not None and args.horizon not in REDUCED_HORIZONS:
            raise ValueError("Horizon khong thuoc 225 runs. Dung --grid full neu muon full grid.")

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs và batch size phải dương")

    if args.max_runs < 0:
        raise ValueError("max-runs không được âm")

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    grid = build_grid(args)

    print(f"Tổng số cấu hình: {len(grid)}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Seed: {config.seed}")

    if not args.execute:
        save_plan(grid, config)
        print("Chế độ lập kế hoạch: chưa huấn luyện")
        return

    completed = 0
    skipped = 0

    for run in grid:
        # Giới hạn số run thực sự huấn luyện mới
        if args.max_runs > 0 and completed >= args.max_runs:
            break

        # Config luu trong Kaggle: M0/M1/M2 lambda=0; M3/M4 lambda=0.1
        run_config = replace(
            config, lambda_align=(0.1 if run["model"] in {"M3", "M4"} else 0.0)
        )
        status = run_experiment(run, run_config)

        if status == "completed":
            completed += 1
        elif status == "paused":
            print("[PAUSED] Dung vong lap de giu checkpoint dang do")
            break
        elif status == "skipped":
            skipped += 1
        else:
            raise RuntimeError(f"Trang thai run khong hop le: {status}")

    print(f"Hoàn thành mới: {completed}")
    print(f"Bỏ qua: {skipped}")


if __name__ == "__main__":
    main()
