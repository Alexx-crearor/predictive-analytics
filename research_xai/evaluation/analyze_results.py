from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from statistics import mean, median

from research_xai.evaluation.metrics import (
    compare_pair,
    relative_improvement_pct,
    summarize_early_stopping,
    summarize_model,
    summarize_pairs,
)

DATASETS = ("Electricity", "Financial", "CMIP6 Wind", "CMIP6 Pressure", "Lorenz")
MODELS = ("M0", "M1", "M2", "M3", "M4")
LOOKBACKS = (70, 100, 130)
HORIZONS = (2, 8, 16)
SEED = 7
ID_FOR_DATASET = {
    "Electricity": "energy", "Financial": "financial", "CMIP6 Wind": "wind",
    "CMIP6 Pressure": "pressure", "Lorenz": "lorenz",
}
PAIRS = (("M1", "M4"), ("M0", "M4"), ("M3", "M4"), ("M1", "M3"), ("M1", "M2"))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = PROJECT_ROOT / "research_xai/results/reduced_grid_225_v1/metrics_merged.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "research_xai/results/reduced_grid_225_v1/analysis"


def load(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy CSV: {path}")
    numeric_float = ("lambda_align", "best_val_mse", "test_mse", "test_mae")
    numeric_int = ("lookback", "horizon", "seed", "best_epoch", "stopped_epoch", "epochs_saved")
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        needed = set(numeric_float + numeric_int + ("dataset", "model", "early_stopped"))
        missing = needed.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV thiếu cột: {sorted(missing)}")
        for line, item in enumerate(reader, start=2):
            try:
                row = dict(item)
                for k in numeric_float:
                    row[k] = float(row[k])
                for k in numeric_int:
                    row[k] = int(row[k])
                raw = row["early_stopped"].strip().lower()
                if raw not in {"true", "false", "1", "0"}:
                    raise ValueError("early_stopped không phải boolean")
                row["early_stopped"] = raw in {"true", "1"}
                if not all(math.isfinite(row[k]) for k in numeric_float):
                    raise ValueError("MSE/MAE/lambda chứa NaN hoặc Inf")
                if min(row["test_mse"], row["test_mae"], row["best_val_mse"]) < 0:
                    raise ValueError("MSE/MAE phải không âm")
                rows.append(row)
            except (ValueError, TypeError) as err:
                raise ValueError(f"CSV dòng {line}: {err}") from err
    return rows


def validate(rows):
    expected = set(product(DATASETS, MODELS, LOOKBACKS, HORIZONS, (SEED,)))
    seen = {}
    for r in rows:
        key = (r["dataset"], r["model"], r["lookback"], r["horizon"], r["seed"])
        if key in seen:
            raise ValueError(f"Cấu hình trùng: {key}")
        if key not in expected:
            raise ValueError(f"Cấu hình ngoài grid: {key}")
        wanted_lambda = 0.1 if r["model"] in ("M3", "M4") else 0.0
        if not math.isclose(r["lambda_align"], wanted_lambda, abs_tol=1e-9):
            raise ValueError(f"Lambda sai cho {key}: {r['lambda_align']}")
        if not (0 < r["best_epoch"] <= r["stopped_epoch"] <= 4000):
            raise ValueError(f"Epoch bất hợp lý: {key}")
        if r["epochs_saved"] != 4000 - r["stopped_epoch"]:
            raise ValueError(f"epochs_saved không khớp: {key}")
        seen[key] = r
    missing = sorted(expected - set(seen))
    if missing:
        raise ValueError(f"Chưa đủ 225 runs: {len(seen)}/225; thiếu {missing[:5]}")
    assert len(seen) == 225
    return seen


def write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path} ({len(rows)} rows)")


def fmt(value, decimals=4):
    return f"{value:.{decimals}f}"


def build_outputs(bykey):
    """Bảng cho báo cáo; mọi phép tính dùng hàm trong `evaluation.metrics`."""
    meanrows = []
    efficiency = []
    horizonrows = []
    winners = []
    paired = []
    summaries = []
    xai = []

    for dataset in DATASETS:
        for model in MODELS:
            records = [bykey[(dataset, model, p, h, SEED)] for p, h in product(LOOKBACKS, HORIZONS)]
            meanrows.append({"dataset": dataset, "model": model, **summarize_model(records)})
            efficiency.append({"dataset": dataset, "model": model,
                               **summarize_early_stopping(records, target_epochs=4000)})
            for h in HORIZONS:
                subset = [bykey[(dataset, model, p, h, SEED)] for p in LOOKBACKS]
                s = summarize_model(subset)
                horizonrows.append({"dataset": dataset, "model": model, "horizon": h,
                                    "runs": s["runs"], "mean_test_mse": s["mean_test_mse"],
                                    "mean_test_mae": s["mean_test_mae"]})

        # 45 tổ hợp dataset/lookback/horizon, mỗi tổ hợp có 5 models
        for p, h in product(LOOKBACKS, HORIZONS):
            records = [bykey[(dataset, model, p, h, SEED)] for model in MODELS]
            best = min(records, key=lambda x: x["test_mse"])
            winners.append({"dataset": dataset, "lookback": p, "horizon": h,
                            "winner": best["model"], "winning_mse": best["test_mse"], "seed": SEED})

        # Paired comparison: cùng dataset, P, H, seed, KHÔNG ghép tùy ý
        for baseline, candidate in PAIRS:
            current_pairs = []
            for p, h in product(LOOKBACKS, HORIZONS):
                b = bykey[(dataset, baseline, p, h, SEED)]
                c = bykey[(dataset, candidate, p, h, SEED)]
                comparison = compare_pair(b, c, baseline, candidate)
                paired.append(comparison)
                current_pairs.append(comparison)
            summaries.append({"dataset": dataset, "baseline": baseline, "candidate": candidate,
                              **summarize_pairs(current_pairs)})

        # Chọn P100/H8 TRƯỚC khi xem test MSE; không cherry-pick XAI
        p, h = 100, 8
        selection = {model: bykey[(dataset, model, p, h, SEED)] for model in MODELS}
        xai.append({
            "dataset": dataset, "dataset_key": ID_FOR_DATASET[dataset],
            "lookback": p, "horizon": h, "seed": SEED,
            "M0_mse": selection["M0"]["test_mse"],
            "M1_mse": selection["M1"]["test_mse"],
            "M2_mse": selection["M2"]["test_mse"],
            "M3_mse": selection["M3"]["test_mse"],
            "M4_mse": selection["M4"]["test_mse"],
            "M4_vs_M1_improvement_pct": relative_improvement_pct(
                selection["M1"]["test_mse"], selection["M4"]["test_mse"]
            ),
            "case_rule": "fixed P100/H8 on all datasets; no test-based selection",
            "checkpoint_folder_example": f"{ID_FOR_DATASET[dataset]}/M4/p100_h8_e4000_b32_s7",
        })

    return meanrows, efficiency, horizonrows, winners, paired, summaries, xai


def charts(out: Path, means, efficiency, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] Chưa có matplotlib; bỏ qua PNG. Cài bằng: pip install matplotlib")
        return

    subset = {(r["dataset"],r["baseline"],r["candidate"]):r for r in summary}
    ratios = [subset[(d,"M1","M4")]["mean_candidate_mse"] /
              subset[(d,"M1","M4")]["mean_baseline_mse"] for d in DATASETS]
    improvements = [100 * (1 - v) for v in ratios]
    fig, ax = plt.subplots(figsize=(10,4.5))
    bars = ax.barh(DATASETS, ratios)
    ax.invert_yaxis()
    ax.axvline(1, linewidth=1, linestyle="--", label="M4 = M1")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10], labels=["0.5×", "1×", "2×", "5×", "10×"])
    ax.set_xlim(0.45, max(12, max(ratios) * 1.6))
    ax.set_xlabel("M4 mean test MSE / M1 mean test MSE (lower = better)")
    ax.set_title("Paired configurations, P={70,100,130}, H={2,8,16}, seed=7")
    for bar,v,imp in zip(bars,ratios,improvements):
        ax.annotate(f"{imp:+.1f}%", (v,bar.get_y()+bar.get_height()/2),
                    xytext=(5,0),textcoords="offset points",va="center",ha="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out/"m4_vs_m1_by_dataset.png",dpi=170)
    plt.close(fig)

    index={(r["dataset"],r["model"]):r for r in means}
    fig,ax=plt.subplots(figsize=(10,5))
    x=list(range(len(DATASETS)))
    for i,m in enumerate(MODELS):
        pos=[j+(i-2)*0.15 for j in x]
        rel=[index[(d,m)]["mean_test_mse"]/index[(d,"M1")]["mean_test_mse"] for d in DATASETS]
        ax.bar(pos,rel,0.15,label=m)
    ax.axhline(1,linewidth=1,linestyle="--")
    ax.set_xticks(x,DATASETS,rotation=20,ha="right")
    ax.set_ylabel("Mean test MSE / M1 mean test MSE (lower = better)")
    ax.set_title("Normalize within each dataset; do not average raw MSE across datasets")
    ax.legend(ncol=5)
    fig.tight_layout()
    fig.savefig(out/"relative_mse_models.png",dpi=170)
    plt.close(fig)

    early={m:mean(r["mean_stopped_epoch"] for r in efficiency if r["model"]==m) for m in MODELS}
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.bar(list(early),list(early.values()))
    ax.axhline(4000,linestyle="--",linewidth=1,label="Max epoch budget")
    ax.set_ylim(0,4400)
    ax.set_ylabel("Mean epochs trained")
    ax.set_title("Early stopping (all 225 runs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"early_stopping_epochs.png",dpi=170)
    plt.close(fig)
    print("[PNG] m4_vs_m1_by_dataset.png, relative_mse_models.png, early_stopping_epochs.png")


def report(out, meanrows, efficiency, winners, summaries, xai):
    sm={(r["dataset"],r["baseline"],r["candidate"]):r for r in summaries}
    means={(r["dataset"],r["model"]):r for r in meanrows}
    lines=[
        "# DeepKoopFormer — 225 runs: kết quả định lượng",
        "",
        "Nguồn: `research_xai/results/reduced_grid_225_v1/metrics_merged.csv`. ",
        "Giao thức: 5 datasets × M0–M4 × P={70,100,130} × H={2,8,16}, seed=7; 4.000 max epochs, batch 32.",
        "",
        "## MSE trung bình theo dataset (9 cấu hình/model)",
        "",
        "| Dataset | M0 | M1 | M2 | M3 | M4 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for d in DATASETS:
        lines.append("| "+d+" | "+" | ".join(f"{means[(d,m)]['mean_test_mse']:.7g}" for m in MODELS)+" |")
    lines += ["", "## M4 so với M1 (9 cặp/dataset)", "",
              "Cột cải thiện được tính bằng `100 × (1 − mean(M4 MSE) / mean(M1 MSE))` trên cùng 9 tổ hợp P/H.",
              "", "| Dataset | M4 thắng M1 | Cải thiện MSE trung bình |", "|---|---:|---:|"]
    for d in DATASETS:
        s=sm[(d,"M1","M4")]
        lines.append(f"| {d} | {s['candidate_wins']}/9 | {s['improvement_of_mean_mse_pct']:+.2f}% |")
    lines += ["", "## M4 so với M3", "", "| Dataset | M4 thắng M3 | Cải thiện MSE trung bình |", "|---|---:|---:|"]
    for d in DATASETS:
        s=sm[(d,"M3","M4")]
        lines.append(f"| {d} | {s['candidate_wins']}/9 | {s['improvement_of_mean_mse_pct']:+.2f}% |")
    avg_epochs=mean(r["mean_stopped_epoch"] for r in efficiency)
    lines += ["", "## Early stopping", "", f"- Trung bình {avg_epochs:.1f}/4000 epochs/run, tiết kiệm {100*(1-avg_epochs/4000):.2f}% ngân sách epochs.",
              f"- Early stopped: {sum(r['early_stopped_count'] for r in efficiency)}/225 runs.",
              "", "## Chọn checkpoint XAI không cherry-pick", "",
              "Giữ **P100/H8** cho tất cả 5 datasets; so sánh M0/M1/M3/M4 trên **cùng test window**.",
              "", "| Dataset | M1 Test MSE | M4 Test MSE | M4 vs M1 |", "|---|---:|---:|---:|"]
    for x in xai:
        lines.append(f"| {x['dataset']} | {x['M1_mse']:.7g} | {x['M4_mse']:.7g} | {x['M4_vs_M1_improvement_pct']:+.1f}% |")
    lines += ["", "## Giới hạn và các kiểm chứng chưa hoàn thành", "",
              "- Kết quả hiện có từ **1 seed**; không coi 9 cấu hình là 9 lần lặp độc lập để khẳng định ý nghĩa thống kê.",
              "- Chỉ đánh giá grid đại diện 225 runs, không phải full grid 1.400 runs.",
              "- CMIP6 Pressure dùng `cmip6_pressure_candidate_2500.pt`, nguồn/split cần xác minh trước khi kết luận chính thức.",
              "- So sánh giữa kỳ cũ cần ghi rõ khác biệt preprocessing, batch, validation và scaling; không gộp MSE trực tiếp.",
              "- XAI/latent diagnostics hiện **chưa thực hiện**; không sử dụng các cơ chế giả định để kết luận nguyên nhân thất bại.",
              "- Không lấy trung bình MSE thô giữa các datasets có thang độ khác nhau; chuẩn hóa theo M1 hoặc phân tích riêng từng dataset.", ""]
    (out/"REPORT_RESULTS_225.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"[MD] {out/'REPORT_RESULTS_225.md'}")


def main():
    p=argparse.ArgumentParser(description="Phân tích 225 runs từ metrics_merged.csv; không train")
    p.add_argument("--metrics",type=Path,default=DEFAULT_METRICS)
    p.add_argument("--out",type=Path,default=DEFAULT_OUTPUT)
    args=p.parse_args()
    rows=load(args.metrics)
    bykey=validate(rows)
    args.out.mkdir(parents=True,exist_ok=True)
    meanrows,efficiency,horizons,winners,paired,summary,xai=build_outputs(bykey)
    write_csv(args.out/"mean_mse_by_dataset_model.csv",meanrows,("dataset","model","runs","mean_test_mse","median_test_mse","mean_test_mae","mean_best_val_mse"))
    write_csv(args.out/"early_stopping_by_dataset_model.csv",efficiency,("dataset","model","runs","early_stopped_count","mean_best_epoch","mean_stopped_epoch","mean_epochs_saved","epoch_budget_saved_pct"))
    write_csv(args.out/"mean_mse_by_horizon.csv",horizons,("dataset","model","horizon","runs","mean_test_mse","mean_test_mae"))
    write_csv(args.out/"winner_per_configuration.csv",winners,("dataset","lookback","horizon","winner","winning_mse","seed"))
    write_csv(args.out/"paired_comparisons.csv",paired,("dataset","baseline","candidate","lookback","horizon","seed","baseline_mse","candidate_mse","absolute_improvement_mse","relative_improvement_pct","candidate_wins"))
    write_csv(args.out/"paired_summary.csv",summary,("dataset","baseline","candidate","n_pairs","candidate_wins","win_rate_pct","mean_baseline_mse","mean_candidate_mse","improvement_of_mean_mse_pct","mean_paired_improvement_pct","median_paired_improvement_pct"))
    write_csv(args.out/"xai_fixed_cases.csv",xai,("dataset","dataset_key","lookback","horizon","seed","M0_mse","M1_mse","M2_mse","M3_mse","M4_mse","M4_vs_M1_improvement_pct","case_rule","checkpoint_folder_example"))
    charts(args.out,meanrows,efficiency,summary)
    report(args.out,meanrows,efficiency,winners,summary,xai)
    print("\nPASS: grid 225/225, all unique, lambda/epochs consistent.")
    print("Results: ",args.out)


if __name__=="__main__":
    main()
