
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research_xai.data.dataset import prepare_dataset
from research_xai.scripts.run_grid import DATASETS, create_model
from research_xai.training.trainer import calculate_data_hash
from research_xai.explainability.integrated_gradients import (
    integrated_gradients,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research_xai/results/reduced_grid_225_v1"


def load_checkpoint(dataset_key, model_name, data, lookback, horizon, device):
    """Nạp đúng checkpoint đã huấn luyện. Không train lại."""

    run_dir = (
        RESULTS / dataset_key / model_name
        / f"p{lookback}_h{horizon}_e4000_b32_s7"
    )

    path = run_dir / "checkpoint_best.pt"
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False
    )

    data_hash = calculate_data_hash(
        data["x_train"], data["y_train"],
        data["x_val"], data["y_val"],
    )

    if checkpoint["data_hash"] != data_hash:
        raise ValueError("Checkpoint không khớp dữ liệu!")

    if checkpoint["variant"] != model_name:
        raise ValueError("Checkpoint không khớp mô hình!")

    model = create_model(
        variant=model_name,
        input_dim=data["input_dim"],
        seq_len=lookback,
        horizon=horizon,
    )

    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device)
    model.eval()

    return model


def run(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 1. Chuẩn bị dữ liệu đúng giao thức 225 runs
    data = prepare_dataset(
        dataset_name=DATASETS[args.dataset],
        seq_len=args.lookback,
        horizon=args.horizon,
    )

    # Cùng một bộ test windows cho tất cả mô hình
    n_test = len(data["x_test"])
    if not 1 <= args.samples <= n_test:
        raise ValueError("Số samples không hợp lệ")

    indices = np.linspace(
        0, n_test - 1, args.samples, dtype=int
    )

    inputs = data["x_test"][indices].to(device)

    # Baseline trung bình từ train, shape (1, 1, C)
    baseline = data["x_train"].mean(
        dim=(0, 1), keepdim=True
    ).to(device)

    print(f"Device: {device}")
    print(f"Test indices: {indices.tolist()}")

    records = []
    step_values = sorted(set(args.steps))
    reference_steps = max(step_values)

    for model_name in args.models:
        model = load_checkpoint(
            args.dataset, model_name, data,
            args.lookback, args.horizon, device,
        )

        # 2. Tính IG ở các mức độ xấp xỉ khác nhau
        results = {}

        for steps in step_values:
            result = integrated_gradients(
                model=model,
                inputs=inputs,
                baseline=baseline,
                horizon=args.horizon,
                target_channel=0,
                target_step=-1,
                steps=steps,
            )
            results[steps] = result

        # 3. IG nhiều bước nhất làm mốc tham chiếu
        reference = results[reference_steps].attribution
        reference_flat = reference.flatten(1)

        for steps in step_values:
            result = results[steps]

            attribution_flat = result.attribution.flatten(1)

            cosine = F.cosine_similarity(
                attribution_flat,
                reference_flat,
                dim=1,
            ).mean().item()

            abs_error = (
                result.completeness_error.abs().mean().item()
            )

            # Chuẩn hóa theo độ thay đổi của dự báo
            target_delta = (
                result.target_at_input
                - result.target_at_baseline
            )

            relative_error = (
                result.completeness_error.abs()
                / target_delta.abs().clamp_min(1e-6)
            ).mean().item()

            row = {
                "dataset": args.dataset,
                "model": model_name,
                "lookback": args.lookback,
                "horizon": args.horizon,
                "samples": args.samples,
                "ig_steps": steps,
                "reference_steps": reference_steps,
                "mean_abs_completeness_error": abs_error,
                "mean_relative_completeness_error": relative_error,
                "cosine_with_reference": cosine,
            }

            records.append(row)

            print(
                f"[{model_name}] steps={steps:3d} "
                f"| error={abs_error:.8f} "
                f"| cosine={cosine:.6f}"
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 4. Lưu kết quả số, chưa trực quan hóa
    output_dir = (
        RESULTS / "xai"
        / f"p{args.lookback}_h{args.horizon}"
        / "validation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"ig_convergence_{args.dataset}.csv"

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[PASS] Saved {len(records)} rows: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="wind", choices=DATASETS)
    parser.add_argument("--models", nargs="+", default=["M0", "M4"])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--steps", nargs="+", type=int,
        default=[8, 16, 32, 64, 128],
    )

    args = parser.parse_args()
    if any(s < 2 for s in args.steps):
        raise ValueError("IG steps phải >= 2")

    run(args)


if __name__ == "__main__":
    main()
