from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from research_xai.data.dataset import prepare_dataset
from research_xai.scripts.run_grid import DATASETS
from research_xai.explainability.ig_convergence import load_checkpoint
from research_xai.explainability.integrated_gradients import target_prediction
from research_xai.explainability.robustness import attribution_robustness


ROOT = Path(__file__).resolve().parents[2]
XAI_ROOT = (
    ROOT / "research_xai/results/reduced_grid_225_v1"
    / "xai/p100_h8"
)

MODELS = ("M0", "M1", "M2", "M3", "M4")
LOOKBACK = 100
HORIZON = 8
IG_STEPS = 32


def read_npz(path):
    """Đọc dữ liệu Phase 1 mà không thay đổi file gốc."""
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as f:
        return {key: f[key].copy() for key in f.files}


def prepare_shared_cases(dataset_key):
    """Tái tạo test set và đối chiếu dữ liệu Phase 1."""
    data = prepare_dataset(
        DATASETS[dataset_key],
        seq_len=LOOKBACK,
        horizon=HORIZON,
    )

    folder = XAI_ROOT / dataset_key
    shared = read_npz(folder / "shared_test_cases.npz")

    indices = shared["sample_indices"].astype(np.int64)

    if (
        indices.ndim != 1
        or len(indices) == 0
        or len(np.unique(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(data["x_test"]))
    ):
        raise ValueError("Test indices không hợp lệ")

    expected_x = data["x_test"][indices].numpy()
    expected_y = data["y_test"][indices].numpy()

    np.testing.assert_allclose(
        shared["x"], expected_x, atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        shared["y"], expected_y, atol=1e-6, rtol=0
    )

    # Baseline phải nhất quán với Phase 1
    train_mean = data["x_train"].mean(dim=(0, 1)).numpy()

    np.testing.assert_allclose(
        shared["baseline_channel"],
        train_mean,
        atol=1e-6,
        rtol=1e-5,
        err_msg="Baseline Phase 1 khác trung bình train",
    )

    return data, folder, shared, indices


def make_noise_bank(x, train_std, dataset_key, args):
    """
    Sinh nhiễu một lần cho mỗi dataset.
    M0-M4 cùng dùng các nhiễu này để so sánh công bằng.
    """
    bank = []
    dataset_number = list(DATASETS).index(dataset_key)

    for alpha_index, alpha in enumerate(args.alphas):
        for repeat in range(args.repeats):
            noise_seed = (
                args.seed
                + 100000 * dataset_number
                + 1000 * alpha_index
                + repeat
            )

            generator = torch.Generator(device="cpu")
            generator.manual_seed(noise_seed)

            standard_noise = torch.randn(
                x.shape,
                generator=generator,
                dtype=x.dtype,
            )

            noise = (
                alpha
                * standard_noise
                * train_std.view(1, 1, -1)
            )

            bank.append({
                "alpha": alpha,
                "repeat": repeat,
                "noise_seed": noise_seed,
                "perturbed_x": x + noise,
            })

    return bank


def run_dataset(dataset_key, args, device):
    data, folder, shared, indices = prepare_shared_cases(
        dataset_key
    )

    # Dùng dữ liệu trên CPU khi tạo nhiễu có seed cố định
    x_cpu = torch.from_numpy(shared["x"])
    x = x_cpu.to(device)

    baseline = torch.from_numpy(
        shared["baseline_channel"]
    ).to(device)

    # Độ lệch chuẩn lấy từ train windows, không dùng test
    train_std = data["x_train"].std(
        dim=(0, 1), unbiased=False
    )

    noise_bank = make_noise_bank(
        x_cpu, train_std, dataset_key, args
    )

    rows = []

    print(
        f"\n[DATASET] {dataset_key} "
        f"| samples={len(indices)} "
        f"| noise settings={len(noise_bank)}"
    )

    for model_name in args.models:
        phase1 = read_npz(
            folder / model_name / "xai_arrays.npz"
        )

        original_ig = torch.from_numpy(
            phase1["ig_attributions"]
        )

        if original_ig.shape != x_cpu.shape:
            raise ValueError(
                f"{model_name}: IG shape không khớp"
            )

        if not torch.isfinite(original_ig).all():
            raise ValueError(
                f"{model_name}: IG chứa NaN/Inf"
            )

        # Nạp checkpoint từ 225 runs; kiểm tra data_hash
        model = load_checkpoint(
            dataset_key,
            model_name,
            data,
            LOOKBACK,
            HORIZON,
            device,
        )

        # Đảm bảo cùng đầu ra dự báo với Phase 1
        with torch.no_grad():
            selected = target_prediction(
                model(x),
                channels=data["input_dim"],
                horizon=HORIZON,
                target_channel=0,
                target_step=-1,
            ).cpu().numpy()

        np.testing.assert_allclose(
            selected,
            phase1["selected_prediction"],
            atol=1e-5,
            rtol=1e-4,
            err_msg=f"Prediction Phase 1 không khớp: {model_name}",
        )

        model_rows = []

        for experiment in noise_bank:
            noisy_x = experiment["perturbed_x"].to(device)

            # Tái sử dụng chính hàm robustness đã test
            result = attribution_robustness(
                model=model,
                inputs=x,
                perturbed_inputs=noisy_x,
                original_ig=original_ig,
                baseline=baseline,
                horizon=HORIZON,
                target_channel=0,
                target_step=-1,
                ig_steps=IG_STEPS,
                block_size=args.block_size,
                top_fraction=args.top_fraction,
            )

            measures = {
                "cosine_similarity": result.cosine_similarity.numpy(),
                "relative_l1_change": result.relative_l1_change.numpy(),
                "topk_overlap": result.topk_overlap.numpy(),
                "prediction_shift": result.prediction_shift.numpy(),
            }

            # Không ghi các kết quả lỗi vào CSV
            for name, values in measures.items():
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"{dataset_key}/{model_name}: {name} không hữu hạn"
                    )

            for j, sample_index in enumerate(indices):
                row = {
                    "dataset": dataset_key,
                    "model": model_name,
                    "sample_index": int(sample_index),
                    "lookback": LOOKBACK,
                    "horizon": HORIZON,
                    "ig_steps": IG_STEPS,
                    "noise_alpha": experiment["alpha"],
                    "repeat": experiment["repeat"],
                    "noise_seed": experiment["noise_seed"],
                    "block_size": args.block_size,
                    "top_fraction": args.top_fraction,
                    **{
                        name: float(values[j])
                        for name, values in measures.items()
                    },
                }

                model_rows.append(row)

        rows.extend(model_rows)

        print(
            f"[PASS] {dataset_key}/{model_name}"
            f" | rows={len(model_rows)}"
            f" | cosine={np.mean([r['cosine_similarity'] for r in model_rows]):.6f}"
            f" | overlap={np.mean([r['topk_overlap'] for r in model_rows]):.6f}"
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Chỉ ghi khi toàn bộ models của dataset đã chạy xong
    output_dir = XAI_ROOT / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / f"robustness_{dataset_key}.csv"
    temp = output.with_suffix(".csv.tmp")

    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    temp.replace(output)

    print(f"[PASS] Saved {len(rows)} rows: {output}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", choices=["all", *DATASETS],
        default="wind"
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODELS,
        default=list(MODELS)
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float,
        default=[0.005, 0.01, 0.02]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--top-fraction", type=float, default=0.2)

    args = parser.parse_args()

    if (
        args.repeats < 1
        or args.block_size < 1
        or not 0 < args.top_fraction <= 1
        or not args.alphas
        or not all(0 < a <= 1 for a in args.alphas)
    ):
        raise ValueError("Tham số robustness không hợp lệ")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    datasets = (
        list(DATASETS)
        if args.dataset == "all"
        else [args.dataset]
    )

    print(f"[ROBUSTNESS] device={device}")

    for dataset_key in datasets:
        run_dataset(dataset_key, args, device)

    print("[DONE] Robustness validation completed")


if __name__ == "__main__":
    main()
