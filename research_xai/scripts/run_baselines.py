"""Train 10 baseline controls (5 datasets x LSTM/DLinear) at P100/H8.

Run from repository root:
    python -m research_xai.scripts.run_baselines --smoke-test
    python -m research_xai.scripts.run_baselines --dataset all --models LSTM DLinear --execute

This script does not modify the existing 225-run metrics or checkpoint directories.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from research_xai.data.dataset import prepare_dataset
from research_xai.evaluation.metrics import evaluate_model
from research_xai.models.baselines import DLinearForecaster, LSTMForecaster
from research_xai.scripts.run_grid import DATASETS
from research_xai.training.trainer import TrainConfig, calculate_data_hash, set_seed, train_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJECT_ROOT / "research_xai" / "results" / "baseline_controls_p100_h8"
P, H, SEED = 100, 8, 7
MODELS = ("LSTM", "DLinear")


def build_model(name: str, channels: int) -> torch.nn.Module:
    if name == "LSTM":
        return LSTMForecaster(channels, P, H, hidden_size=96, num_layers=1)
    if name == "DLinear":
        return DLinearForecaster(channels, P, H, moving_avg=25)
    raise ValueError(f"Unknown baseline: {name}")


def run_one(dataset_key: str, name: str, args, config: TrainConfig):
    data = prepare_dataset(DATASETS[dataset_key], seq_len=P, horizon=H)
    data_hash = calculate_data_hash(data["x_train"], data["y_train"],
                                    data["x_val"], data["y_val"])
    dest = args.out / dataset_key / name / f"p{P}_h{H}_e{config.epochs}_b{config.batch_size}_s{config.seed}"
    metrics_path = dest / "metrics.json"
    best_path = dest / "checkpoint_best.pt"
    if metrics_path.exists():
        meta = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (not best_path.exists() or meta.get("status") != "completed" or
            meta.get("data_hash") != data_hash or
            meta.get("config") != asdict(config) or
            meta.get("model") != name):
            raise RuntimeError(f"Completed run mismatch; refusing overwrite: {dest}")
        print(f"[SKIP] {dataset_key}/{name}: verified completed run")
        return meta

    set_seed(SEED)  # Must be BEFORE model creation, as in run_grid.
    model = build_model(name, data["input_dim"])
    n_params = sum(p.numel() for p in model.parameters())
    assert model(data["x_train"][:2]).shape == (2, data["input_dim"] * H)
    print(f"[RUN] {dataset_key}/{name} | train={len(data['x_train'])} "
          f"val={len(data['x_val'])} test={len(data['x_test'])} | params={n_params}", flush=True)

    if args.smoke_test:
        return {"dataset_key": dataset_key, "model": name, "smoke_test": True,
                "params": n_params, "data_hash": data_hash}
    if not args.execute:
        return {"dataset_key": dataset_key, "model": name, "planned": True,
                "params": n_params, "data_hash": data_hash}

    training = train_model(model, name, data["x_train"], data["y_train"],
                           data["x_val"], data["y_val"], dest,
                           config=config, resume=True)
    if training.get("time_limited"):
        print(f"[PAUSED] {dataset_key}/{name}; resume later")
        return None
    if not best_path.is_file():
        raise RuntimeError(f"Missing best checkpoint: {best_path}")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if (best["variant"] != name or best["data_hash"] != data_hash or
        best["config"] != asdict(config)):
        raise RuntimeError("Best checkpoint metadata mismatch")
    model.load_state_dict(best["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    scores = evaluate_model(model, data["x_test"], data["y_test"],
                            batch_size=config.batch_size)
    result = {"dataset": DATASETS[dataset_key], "dataset_key": dataset_key,
              "model": name, "patch": P, "horizon": H, "seed": SEED,
              "params": n_params, "status": "completed", "data_hash": data_hash,
              "config": asdict(config), "target_epochs": config.epochs,
              "completed_epochs": training["last_epoch"],
              "epochs_saved": training["epochs_saved"],
              "early_stopped": training["early_stopped"],
              "best_epoch": training["best_epoch"],
              "best_val_mse": training["best_val_mse"], **scores}
    dest.mkdir(parents=True, exist_ok=True)
    tmp = metrics_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    tmp.replace(metrics_path)
    print(f"[PASS] {dataset_key}/{name} MSE={scores['test_mse']:.8g} "
          f"MAE={scores['test_mae']:.8g}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    parser.add_argument("--execute", action="store_true",
                        help="Actually train; otherwise only validate data and model shapes")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.execute and args.smoke_test:
        parser.error("Choose --execute OR --smoke-test")

    config = replace(TrainConfig(seed=SEED), lambda_align=0.0)
    selection = list(DATASETS) if args.dataset == "all" else [args.dataset]
    args.out = args.out.resolve()
    rows = []
    for dataset in selection:
        for name in dict.fromkeys(args.models):
            result = run_one(dataset, name, args, config)
            if result is not None and result.get("status") == "completed":
                rows.append(result)

    if args.execute:
        # Aggregate only successfully completed runs, including previous sessions.
        rows = []
        for dataset in DATASETS:
            for name in MODELS:
                path = args.out / dataset / name / f"p{P}_h{H}_e{config.epochs}_b{config.batch_size}_s{config.seed}" / "metrics.json"
                if path.is_file():
                    meta = json.loads(path.read_text(encoding="utf-8"))
                    if meta.get("status") == "completed":
                        rows.append({k: meta[k] for k in
                                     ("dataset", "dataset_key", "model", "patch", "horizon", "seed",
                                      "params", "test_mse", "test_mae", "best_epoch", "completed_epochs",
                                      "epochs_saved", "early_stopped", "data_hash")})
        if rows:
            args.out.mkdir(parents=True, exist_ok=True)
            summary = args.out / "baseline_summary.csv"
            with summary.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[SUMMARY] {summary} | completed={len(rows)}/10")
    else:
        print("[DRY RUN] No weights or metrics were written")


if __name__ == "__main__":
    main()
