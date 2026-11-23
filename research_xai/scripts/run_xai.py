from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from research_xai.data.dataset import prepare_dataset
from research_xai.models.forecasters import PatchTSTDirect, KoopmanDirect, TemporalKoopman
from research_xai.training.trainer import calculate_data_hash

from research_xai.explainability.integrated_gradients import integrated_gradients, target_prediction
from research_xai.explainability.temporal_occlusion import temporal_occlusion
from research_xai.explainability.koopman_attribution import koopman_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "research_xai" / "results" / "reduced_grid_225_v1"
DATASETS = {
    "energy": "Electricity", "financial": "Financial", "wind": "CMIP6 Wind",
    "pressure": "CMIP6 Pressure", "lorenz": "Lorenz",
}
MODELS = ("M0", "M1", "M2", "M3", "M4")
LOOKBACK = 100
HORIZON = 8
SEED = 7


def get_model(variant: str, channels: int) -> torch.nn.Module:
    """Giữ chính xác hyperparameters kiến trúc trong research_xai.scripts.run_grid.create_model."""
    common = dict(input_dim=channels, seq_len=LOOKBACK, horizon=HORIZON,
                  patch_len=LOOKBACK, d_model=96)
    if variant == "M0":
        return PatchTSTDirect(**common)
    if variant == "M1":
        return KoopmanDirect(**common, p_max=0.99)
    if variant in ("M2", "M3", "M4"):
        return TemporalKoopman(**common, p_max=0.99, affine=(variant == "M4"))
    raise ValueError(f"Không nhận diện variant {variant}")


def load_verified_model(
    variant: str, dataset_key: str, data: dict,
    data_hash: str, root: Path, device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    run_dir = root / dataset_key / variant / f"p{LOOKBACK}_h{HORIZON}_e4000_b32_s{SEED}"
    metrics_file = run_dir / "metrics.json"
    weights_file = run_dir / "checkpoint_best.pt"
    if not metrics_file.is_file() or not weights_file.is_file():
        raise FileNotFoundError(f"Thiếu metrics/checkpoint: {run_dir}")
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    required_lambda = 0.1 if variant in ("M3", "M4") else 0.0
    if not (
        metrics.get("status") == "completed"
        and metrics.get("dataset_key") == dataset_key
        and metrics.get("model") == variant
        and metrics.get("patch") == LOOKBACK
        and metrics.get("horizon") == HORIZON
        and metrics.get("seed") == SEED
        and metrics.get("data_hash") == data_hash
        and metrics.get("config", {}).get("epochs") == 4000
        and metrics.get("config", {}).get("batch_size") == 32
        and math.isclose(float(metrics.get("config", {}).get("lambda_align", -1)), required_lambda, abs_tol=1e-10)
    ):
        raise RuntimeError(f"Metadata sai hoặc data hash khác lần train: {run_dir}")

    # Chỉ load checkpoint của dự án tin cậy do bạn tự huấn luyện.
    checkpoint = torch.load(weights_file, map_location="cpu", weights_only=False)
    if not (
        checkpoint.get("variant") == variant
        and checkpoint.get("data_hash") == data_hash
        and checkpoint.get("epoch") == metrics.get("best_epoch")
        and checkpoint.get("config") == metrics.get("config")
    ):
        raise RuntimeError(f"Checkpoint không khớp metrics: {run_dir}")
    model = get_model(variant, data["input_dim"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, metrics


def select_indices(n_total: int, n_samples: int) -> np.ndarray:
    """Chọn trước n windows cách đều, không chọn theo sai số dự báo."""
    if n_total < 1 or n_samples < 1:
        raise ValueError("n_total/n_samples phải dương")
    n = min(n_total, n_samples)
    return np.linspace(0, n_total - 1, num=n, dtype=np.int64)


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="XAI cho DeepKoopFormer P100/H8; chỉ inference")
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="wind")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["M0", "M1", "M3", "M4"])
    parser.add_argument("--methods", nargs="+", choices=("ig", "occlusion", "dynamics"),
                        default=["ig", "occlusion", "dynamics"])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--target-channel", type=int, default=0)
    parser.add_argument("--target-step", type=int, default=-1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.samples < 1 or args.ig_steps < 2 or args.block_size < 1:
        parser.error("samples>=1; ig-steps>=2; block-size>=1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("Không có CUDA GPU")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    out_root = args.out if args.out is not None else args.results_root / "xai" / "p100_h8"
    dataset_keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    model_names = list(dict.fromkeys(args.models))
    methods = set(args.methods)
    summary_rows = []
    print(f"[XAI] device={device}; datasets={dataset_keys}; models={model_names}; methods={sorted(methods)}", flush=True)

    for dataset_key in dataset_keys:
        dataset_name = DATASETS[dataset_key]
        data = prepare_dataset(dataset_name, seq_len=LOOKBACK, horizon=HORIZON)
        data_hash = calculate_data_hash(data["x_train"], data["y_train"], data["x_val"], data["y_val"])
        ids = select_indices(len(data["x_test"]), args.samples)
        x = data["x_test"][ids].float().to(device)
        y = data["y_test"][ids].float().to(device)
        if not 0 <= args.target_channel < data["input_dim"]:
            raise ValueError(f"Dataset {dataset_name}: target_channel={args.target_channel} không hợp lệ")
        # Trung bình trên train windows: same scaler as original model training.
        baseline_channel = data["x_train"].float().mean(dim=(0, 1)).to(device)
        baseline = baseline_channel.view(1, 1, -1)
        selected_step = args.target_step % HORIZON
        if not (-HORIZON <= args.target_step < HORIZON):
            raise ValueError("target-step vượt số bước dự báo")
        source_hash = hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
        ds_out = out_root / dataset_key
        ds_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[DATASET] {dataset_name}; test windows={len(data['x_test'])}; indices={ids.tolist()}", flush=True)

        # Test windows và baseline được lưu một lần, dùng chung cho tất cả model.
        np.savez_compressed(ds_out / "shared_test_cases.npz", sample_indices=ids,
                            x=x.detach().cpu().numpy(), y=y.detach().cpu().numpy(),
                            baseline_channel=baseline_channel.cpu().numpy())

        for variant in model_names:
            model, metadata = load_verified_model(variant, dataset_key, data, data_hash, args.results_root, device)
            run_out = ds_out / variant
            run_out.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                prediction = model(x)
                selected_prediction = target_prediction(
                    prediction, channels=data["input_dim"], horizon=HORIZON,
                    target_channel=args.target_channel, target_step=args.target_step,
                ).detach().cpu().numpy()
                forecast_np = prediction.detach().cpu().numpy().reshape(len(ids), data["input_dim"], HORIZON)
                selected_true = y.reshape(len(ids), data["input_dim"], HORIZON)[:, args.target_channel, selected_step].cpu().numpy()

            row = {
                "dataset": dataset_name, "model": variant, "lookback": LOOKBACK, "horizon": HORIZON,
                "seed": SEED, "n_samples": len(ids), "target_channel": args.target_channel,
                "target_step_zero_based": selected_step,
                "selected_windows_mse": float(np.mean((selected_prediction - selected_true) ** 2)),
                "full_test_mse_from_training": float(metadata["test_mse"]),
                "ig_mean_abs": "", "ig_mean_abs_completeness_error": "",
                "occlusion_mean_abs": "", "ig_occlusion_pearson_abs": "",
                "spectral_radius": "", "sigma_max": "", "shift_gap_mean": "",
                "model_best_epoch": int(metadata["best_epoch"]),
            }
            arrays = dict(prediction=forecast_np, selected_prediction=selected_prediction,
                          selected_true=selected_true)
            ig, occ, dyn = None, None, None

            if "ig" in methods:
                # IG cần autograd theo input, không bật torch.no_grad tại đây.
                ig = integrated_gradients(
                    model, x, baseline, horizon=HORIZON,
                    target_channel=args.target_channel, target_step=args.target_step,
                    steps=args.ig_steps,
                )
                arrays.update(ig_attributions=ig.attribution.numpy(),
                              ig_completeness_error=ig.completeness_error.numpy(),
                              ig_input_prediction=ig.target_at_input.numpy(),
                              ig_baseline_prediction=ig.target_at_baseline.numpy())
                row["ig_mean_abs"] = float(ig.attribution.abs().mean().item())
                row["ig_mean_abs_completeness_error"] = float(ig.completeness_error.abs().mean().item())

            if "occlusion" in methods:
                occ = temporal_occlusion(
                    model, x, baseline, horizon=HORIZON,
                    target_channel=args.target_channel, target_step=args.target_step,
                    block_size=args.block_size,
                )
                arrays["occlusion_impacts"] = occ.impacts.numpy()
                row["occlusion_mean_abs"] = float(occ.impacts.abs().mean().item())

            if ig is not None and occ is not None:
                a = ig.attribution.abs().mean(dim=0).numpy().reshape(-1)
                b = occ.impacts.abs().mean(dim=0).numpy().reshape(-1)
                if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                    row["ig_occlusion_pearson_abs"] = float(np.corrcoef(a, b)[0, 1])

            if "dynamics" in methods:
                dyn = koopman_diagnostics(model, x, y, horizon=HORIZON, variant=variant)
                if dyn is not None:
                    arrays.update(koop_singular_values=dyn.singular_values.numpy(),
                                  koop_centered_norm=dyn.centered_norm.numpy(),
                                  koop_states=dyn.states.numpy())
                    if dyn.equilibrium is not None:
                        arrays["koop_equilibrium"] = dyn.equilibrium.numpy()
                    if dyn.shift_gap is not None:
                        arrays.update(koop_shift_gap=dyn.shift_gap.numpy(),
                                      koop_relative_shift_gap=dyn.relative_shift_gap.numpy())
                        row["shift_gap_mean"] = float(np.nanmean(dyn.shift_gap.numpy()))
                    row["spectral_radius"] = float(dyn.spectral_radius)
                    row["sigma_max"] = float(dyn.singular_values.max().item())

            np.savez_compressed(run_out / "xai_arrays.npz", **arrays)
            metadata_xai = {
                "source_data_hash": data_hash, "selected_test_inputs_sha256": source_hash,
                "method": sorted(methods), "trained_model": variant,
                "dataset": dataset_name, "sample_indices": ids.tolist(),
                "ig_steps": args.ig_steps if ig is not None else None,
                "occlusion_block_size": args.block_size if occ is not None else None,
                "baseline": "mean of normalized x_train windows over samples and timesteps, per channel",
                "target_definition": "forecast[channel, lead] indexed in feature-major output",
                "target_channel": args.target_channel, "target_step_zero_based": selected_step,
                "koopman_gap_definition": "student re-encoding gap, not EMA teacher alignment loss",
                "source_checkpoint": str(args.results_root / dataset_key / variant /
                                        f"p{LOOKBACK}_h{HORIZON}_e4000_b32_s{SEED}" / "checkpoint_best.pt"),
                "source_best_epoch": int(metadata["best_epoch"]),
                "source_full_test_mse": float(metadata["test_mse"]),
                "limitation": "single seed; overlapping test windows; attribution is not causal importance",
            }
            (run_out / "xai_metadata.json").write_text(
                json.dumps(metadata_xai, indent=2, ensure_ascii=False), encoding="utf-8")
            summary_rows.append(row)
            print(f"[PASS] {dataset_key}/{variant}: IG={row['ig_mean_abs']}, "
                  f"Occlusion={row['occlusion_mean_abs']}, sigma={row['sigma_max']}", flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_csv(out_root / "xai_summary.csv", summary_rows)
    print(f"\nPASS: {len(summary_rows)} checkpoint models; data-hash + strict load; output at {out_root}", flush=True)
    print("Để dành visualization Notebook bước cuối, chưa tạo PNG.", flush=True)


if __name__ == "__main__":
    main()
