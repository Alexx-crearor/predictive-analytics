from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from research_xai.data.dataset import prepare_dataset
from research_xai.scripts.run_grid import DATASETS
from research_xai.explainability.ig_convergence import (
    load_checkpoint,
)
from research_xai.explainability.faithfulness import (
    faithfulness_test,
)
from research_xai.explainability.integrated_gradients import (
    target_prediction,
)


ROOT = Path(__file__).resolve().parents[2]
XAI_ROOT = (
    ROOT / "research_xai/results/reduced_grid_225_v1"
    / "xai/p100_h8"
)

MODELS = ("M0", "M1", "M2", "M3", "M4")
LOOKBACK = 100
HORIZON = 8


def read_npz(path):
    """Đọc các mảng đã lưu ở Phase 1."""
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def run_dataset(dataset_key, models, device):
    dataset_name = DATASETS[dataset_key]

    # Bước 1: chuẩn bị lại đúng test set cũ
    data = prepare_dataset(
        dataset_name=dataset_name,
        seq_len=LOOKBACK,
        horizon=HORIZON,
    )

    folder = XAI_ROOT / dataset_key
    shared = read_npz(folder / "shared_test_cases.npz")

    indices = shared["sample_indices"].astype(np.int64)
    x_saved = shared["x"]
    y_saved = shared["y"]
    baseline_saved = shared["baseline_channel"]

    # Bước 2: kiểm tra dữ liệu Phase 1 khớp tập test
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("sample_indices không hợp lệ")

    if np.any(indices < 0) or np.any(indices >= len(data["x_test"])):
        raise ValueError("sample_indices vượt ngoài tập test")

    x_test = data["x_test"][indices].numpy()
    y_test = data["y_test"][indices].numpy()

    if not np.allclose(x_saved, x_test, atol=1e-6, rtol=0):
        raise ValueError("Phase 1 inputs khác test set")

    if not np.allclose(y_saved, y_test, atol=1e-6, rtol=0):
        raise ValueError("Phase 1 targets khác test set")

    channels = data["input_dim"]
    if baseline_saved.shape != (channels,):
        raise ValueError("Baseline không khớp số channels")

    if not np.isfinite(baseline_saved).all():
        raise ValueError("Baseline có NaN/Inf")

    inputs = torch.from_numpy(x_saved).to(device)
    baseline = torch.from_numpy(baseline_saved).to(device)

    rows = []

    # Bước 3: chạy lần lượt từng checkpoint
    for model_name in models:
        saved = read_npz(
            folder / model_name / "xai_arrays.npz"
        )

        ig_array = saved["ig_attributions"]

        if ig_array.shape != x_saved.shape:
            raise ValueError(
                f"{model_name}: IG không khớp input shape"
            )

        if not np.isfinite(ig_array).all():
            raise ValueError(
                f"{model_name}: IG có NaN/Inf"
            )

        model = load_checkpoint(
            dataset_key,
            model_name,
            data,
            LOOKBACK,
            HORIZON,
            device,
        )

        # Bước 4: đối chiếu dự báo với Phase 1
        with torch.no_grad():
            current_prediction = target_prediction(
                model(inputs),
                channels=channels,
                horizon=HORIZON,
                target_channel=0,
                target_step=-1,
            ).cpu().numpy()

        if not np.allclose(
            current_prediction,
            saved["selected_prediction"],
            atol=1e-5,
            rtol=1e-4,
        ):
            raise ValueError(
                f"{model_name}: prediction khác Phase 1"
            )

        # Bước 5: đo faithfulness
        result = faithfulness_test(
            model=model,
            inputs=inputs,
            baseline=baseline,
            ig_attribution=torch.from_numpy(
                ig_array
            ).to(device),
            horizon=HORIZON,
            target_channel=0,
            target_step=-1,
            block_size=10,
            top_fraction=0.2,
            random_repeats=10,
            seed=7,
        )

        top = result.top_delta.numpy()
        bottom = result.bottom_delta.numpy()
        random = result.random_delta.numpy()

        for j, sample_index in enumerate(indices):
            rows.append({
                "dataset": dataset_key,
                "model": model_name,
                "sample_index": int(sample_index),
                "lookback": LOOKBACK,
                "horizon": HORIZON,
                "target_channel": 0,
                "target_step": HORIZON - 1,
                "masked_fraction": 0.2,
                "units_masked": result.units_masked,
                "total_units": result.total_units,
                "top_delta": float(top[j]),
                "bottom_delta": float(bottom[j]),
                "random_delta": float(random[j]),
                "top_minus_bottom": float(
                    top[j] - bottom[j]
                ),
                "top_minus_random": float(
                    top[j] - random[j]
                ),
            })

        print(
            f"[PASS] {dataset_key}/{model_name} "
            f"| samples={len(indices)} "
            f"| top={top.mean():.6f} "
            f"| bottom={bottom.mean():.6f} "
            f"| random={random.mean():.6f}"
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Bước 6: lưu kết quả số
    output_dir = XAI_ROOT / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir / f"faithfulness_{dataset_key}.csv"
    )

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[PASS] Saved {len(rows)} rows: {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS],
        default="wind",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    datasets = (
        list(DATASETS) if args.dataset == "all"
        else [args.dataset]
    )

    print(f"[FAITHFULNESS] device={device}")

    for dataset_key in datasets:
        run_dataset(dataset_key, args.models, device)

    print("[DONE] Faithfulness evaluation completed")


if __name__ == "__main__":
    main()
