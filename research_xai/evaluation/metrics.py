from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class RegressionAccumulator:
    """Tích lũy sai số theo phần tử qua nhiều minibatch."""

    squared_error: float = 0.0
    absolute_error: float = 0.0
    n_values: int = 0

    @torch.no_grad()
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")
        if predictions.numel() == 0:
            raise ValueError("Batch rỗng")
        if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
            raise ValueError("Dự báo hoặc nhãn chứa NaN/Inf")
        errors = predictions - targets
        self.squared_error += errors.square().sum().item()
        self.absolute_error += errors.abs().sum().item()
        self.n_values += errors.numel()

    def compute(self) -> dict[str, float]:
        if self.n_values == 0:
            raise ValueError("Chưa có dữ liệu để đánh giá")
        return {
            "test_mse": self.squared_error / self.n_values,
            "test_mae": self.absolute_error / self.n_values,
        }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 32,
) -> dict[str, float]:
    """Đánh giá một checkpoint trên tập test, KHÔNG cập nhật trọng số.

    Giống giao thức trong `research_xai/run_grid.py` đã chạy trên Kaggle:
    DataLoader không shuffle, lỗi tích lũy trên từng phần tử, model.eval().
    """
    if batch_size <= 0:
        raise ValueError("batch_size phải dương")
    if x_test.size(0) != y_test.size(0) or x_test.size(0) == 0:
        raise ValueError("Số lượng x_test/y_test không hợp lệ")
    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    totals = RegressionAccumulator()
    for inputs, targets in loader:
        predictions = model(inputs.to(device))
        totals.update(predictions, targets.to(device))
    return totals.compute()


def _mean_key(records: list[Mapping[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in records]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError(f"Cột {key} rỗng hoặc chứa giá trị không hợp lệ")
    return mean(values)


def summarize_model(records: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """Tổng hợp 9 cấu hình P/H CỦA CÙNG MỘT dataset/model.

    Không gộp raw MSE từ các dataset khác nhau vì thang đo khác nhau.
    """
    rows = list(records)
    if not rows:
        raise ValueError("Nhóm không có run")
    return {
        "runs": len(rows),
        "mean_test_mse": _mean_key(rows, "test_mse"),
        "median_test_mse": median(float(r["test_mse"]) for r in rows),
        "mean_test_mae": _mean_key(rows, "test_mae"),
        "mean_best_val_mse": _mean_key(rows, "best_val_mse"),
    }


def relative_improvement_pct(baseline: float, candidate: float) -> float:
    """100 * (MSE_baseline - MSE_candidate) / MSE_baseline.

    Dương: candidate tốt hơn; âm: candidate tệ hơn.
    """
    if not all(math.isfinite(float(x)) for x in (baseline, candidate)):
        raise ValueError("MSE phải hữu hạn")
    if baseline <= 0 or candidate < 0:
        raise ValueError("baseline MSE phải > 0, candidate MSE phải >= 0")
    return 100.0 * (baseline - candidate) / baseline


def compare_pair(
    base_row: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
    baseline: str,
    candidate: str,
) -> dict[str, Any]:
    """So sánh hai mô hình tại CÙNG dataset/lookback/horizon/seed."""
    keycols = ("dataset", "lookback", "horizon", "seed")
    if any(base_row[k] != candidate_row[k] for k in keycols):
        raise ValueError("So sánh cặp yêu cầu cùng dataset/lookback/horizon/seed")
    if base_row["model"] != baseline or candidate_row["model"] != candidate:
        raise ValueError("Tên mô hình trong cặp không khớp")
    b = float(base_row["test_mse"])
    c = float(candidate_row["test_mse"])
    return {
        **{k: base_row[k] for k in keycols},
        "baseline": baseline,
        "candidate": candidate,
        "baseline_mse": b,
        "candidate_mse": c,
        "absolute_improvement_mse": b - c,
        "relative_improvement_pct": relative_improvement_pct(b, c),
        "candidate_wins": int(c < b),  # hòa KHÔNG được tính là thắng
    }


def summarize_pairs(records: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """Hai cách tổng hợp phải tách bạch:

    - improvement_of_mean_mse_pct = 100 * (1 - mean(MSE_C)/mean(MSE_B))
    - mean_paired_improvement_pct = mean(100 * (MSE_B - MSE_C)/MSE_B)
    Cả hai đều được công bố để không nhập nhằng phương pháp tính.
    """
    pairs = list(records)
    if not pairs:
        raise ValueError("Không có cặp mô hình")
    baseline = _mean_key(pairs, "baseline_mse")
    candidate = _mean_key(pairs, "candidate_mse")
    wins = sum(int(p["candidate_wins"]) for p in pairs)
    return {
        "n_pairs": len(pairs),
        "candidate_wins": wins,
        "win_rate_pct": 100.0 * wins / len(pairs),
        "mean_baseline_mse": baseline,
        "mean_candidate_mse": candidate,
        "improvement_of_mean_mse_pct": relative_improvement_pct(baseline, candidate),
        "mean_paired_improvement_pct": _mean_key(pairs, "relative_improvement_pct"),
        "median_paired_improvement_pct": median(float(p["relative_improvement_pct"]) for p in pairs),
    }


def summarize_early_stopping(
    records: Iterable[Mapping[str, Any]], target_epochs: int = 4000
) -> dict[str, float | int]:
    """Hiệu quả theo ngân sách EPOCH, không phải thời gian GPU thực tế."""
    rows = list(records)
    if not rows or target_epochs <= 0:
        raise ValueError("Nhóm rỗng hoặc target_epochs không hợp lệ")
    for r in rows:
        stopped = int(r["stopped_epoch"])
        saved = int(r["epochs_saved"])
        if not 0 < stopped <= target_epochs or stopped + saved != target_epochs:
            raise ValueError("stopped_epoch và epochs_saved không khớp")
    mean_saved = _mean_key(rows, "epochs_saved")
    return {
        "runs": len(rows),
        "early_stopped_count": sum(bool(r["early_stopped"]) for r in rows),
        "mean_best_epoch": _mean_key(rows, "best_epoch"),
        "mean_stopped_epoch": _mean_key(rows, "stopped_epoch"),
        "mean_epochs_saved": mean_saved,
        "epoch_budget_saved_pct": 100.0 * mean_saved / target_epochs,
    }
