"""Gộp kết quả reduced grid 225 từ nhiều phiên Kaggle.

Ví dụ:
  python merge_deepkoopformer_225.py --archive-dir kaggle_exports --dry-run
  python merge_deepkoopformer_225.py --archive-dir kaggle_exports --out research_xai/results/reduced_grid_225_v1 --include-best
"""
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

DATASETS = {
    "energy": "Electricity",
    "financial": "Financial",
    "wind": "CMIP6 Wind",
    "pressure": "CMIP6 Pressure",
    "lorenz": "Lorenz",
}
MODELS = ("M0", "M1", "M2", "M3", "M4")
LOOKBACKS = (70, 100, 130)
HORIZONS = (2, 8, 16)
RUN_RE = re.compile(
    r"^research_xai/results/reduced_grid_225_v1/"
    r"(energy|financial|wind|pressure|lorenz)/(M[0-4])/"
    r"p(70|100|130)_h(2|8|16)_e4000_b32_s7/"
    r"(metrics\.json|checkpoint_best\.pt|checkpoint_latest\.pt)$"
)
COLS = (
    "dataset", "model", "lambda_align", "lookback", "horizon", "seed",
    "best_epoch", "stopped_epoch", "epochs_saved", "early_stopped",
    "best_val_mse", "test_mse", "test_mae",
)


def expected_keys():
    return {
        (ds, model, lookback, horizon, 7)
        for ds in DATASETS for model in MODELS
        for lookback in LOOKBACKS for horizon in HORIZONS
    }


def key_text(key):
    ds, model, p, h, seed = key
    return f"{ds}/{model}/p{p}_h{h}_e4000_b32_s{seed}"


def locate_archives(args):
    items = [Path(s) for s in (args.archives or [])]
    if args.archive_dir:
        items += sorted(Path(args.archive_dir).rglob("deepkoopformer_*_gpu*_results.zip"))
    items = sorted(set(items))
    if not items:
        raise SystemExit("Chưa tìm thấy ZIP. Dùng --archive-dir hoặc --archives.")
    for p in items:
        if not p.is_file():
            raise FileNotFoundError(p)
    return items


def collect(archives):
    candidates = defaultdict(list)
    signatures = {}
    source_counts = Counter()
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            broken = z.testzip()
            if broken:
                raise ValueError(f"ZIP hỏng: {archive.name}, entry={broken}")
            metadata = next((n for n in z.namelist() if n.endswith("reduced_grid_runtime_info.json")), None)
            if metadata:
                info = json.loads(z.read(metadata))
                for k in ("git_commit", "trainer_sha256", "runner_sha256", "experiment"):
                    value = info.get(k)
                    if k in signatures and signatures[k] != value:
                        raise ValueError(f"Phiên chạy khác {k}: {archive.name}")
                    signatures[k] = value
            groups = defaultdict(dict)
            for name in z.namelist():
                match = RUN_RE.fullmatch(name)
                if not match:
                    continue
                ds, model, p, h, kind = match.groups()
                key = (ds, model, int(p), int(h), 7)
                groups[key][kind] = name
            source_counts[archive.name] += len(groups)
            for key, files in groups.items():
                if "metrics.json" not in files:
                    candidates[key].append({"archive": archive, "files": files, "metrics": None})
                    continue
                m = json.loads(z.read(files["metrics.json"]))
                expected_lambda = 0.1 if key[1] in ("M3", "M4") else 0.0
                valid = (
                    m.get("status") == "completed"
                    and m.get("dataset_key") == key[0]
                    and m.get("dataset") == DATASETS[key[0]]
                    and m.get("model") == key[1]
                    and m.get("patch") == key[2]
                    and m.get("horizon") == key[3]
                    and m.get("seed") == key[4]
                    and m.get("target_epochs") == 4000
                    and m.get("config", {}).get("epochs") == 4000
                    and m.get("config", {}).get("batch_size") == 32
                    and m.get("config", {}).get("seed") == 7
                    and math.isclose(m.get("config", {}).get("lambda_align", -1), expected_lambda, abs_tol=1e-9)
                    and all(math.isfinite(float(m[v])) for v in ("test_mse", "test_mae", "best_val_mse"))
                    and "checkpoint_best.pt" in files
                    and "checkpoint_latest.pt" in files
                )
                if not valid:
                    raise ValueError(f"Metrics/config không hợp lệ: {archive.name}, {key_text(key)}")
                candidates[key].append({"archive": archive, "files": files, "metrics": m})
    return candidates, signatures, source_counts


def entry_digest(candidate, filename):
    digest = hashlib.sha256()
    with zipfile.ZipFile(candidate["archive"]) as z:
        with z.open(candidate["files"][filename]) as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def choose(candidates):
    completed = {}
    paused = {}
    for key, options in candidates.items():
        dones = [o for o in options if o["metrics"] is not None]
        if not dones:
            paused[key] = max(options, key=lambda o: len(o["files"]))
            continue
        if len(dones) > 1:
            first = dones[0]["metrics"]
            for other in dones[1:]:
                if other["metrics"] != first or (
                    entry_digest(other, "checkpoint_best.pt") !=
                    entry_digest(dones[0], "checkpoint_best.pt")
                ):
                    raise ValueError(f"Có hai checkpoint hoàn thành mâu thuẫn: {key_text(key)}")
        completed[key] = dones[0]
    return completed, paused


def extract_entry(candidate, filename, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(candidate["archive"]) as z:
        with z.open(candidate["files"][filename]) as source, dest.open("wb") as sink:
            shutil.copyfileobj(source, sink, length=1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Gộp 225 cấu hình DeepKoopFormer từ Kaggle")
    parser.add_argument("--archive-dir", help="Thư mục chứa các ZIP từ Kaggle")
    parser.add_argument("--archives", nargs="+", help="Danh sách ZIP cụ thể")
    parser.add_argument("--out", default="research_xai/results/reduced_grid_225_v1")
    parser.add_argument("--include-best", action="store_true", help="Giải nén checkpoint_best.pt để làm XAI")
    parser.add_argument("--include-latest", action="store_true", help="Giải nén thêm checkpoint_latest.pt để resume")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra, không ghi kết quả")
    args = parser.parse_args()

    archives = locate_archives(args)
    candidates, signatures, source_counts = collect(archives)
    completed, paused = choose(candidates)
    expected = expected_keys()
    extra = set(candidates) - expected
    missing = sorted(expected - set(completed))
    assert not extra and len(expected) == 225

    print(f"Archives: {len(archives)} | Completed: {len(completed)}/225 | Missing: {len(missing)}")
    print(f"Paused checkpoints: {len(paused)} | Unique configs: {len(candidates)}")
    print("Source signatures:", signatures)
    if args.dry_run:
        for key in sorted(paused):
            print("PAUSED", key_text(key))
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = []
    for key, item in sorted(completed.items()):
        m = item["metrics"]
        lam = m["config"]["lambda_align"]
        rows.append({
            "dataset": m["dataset"], "model": m["model"], "lambda_align": lam,
            "lookback": m["patch"], "horizon": m["horizon"], "seed": m["seed"],
            "best_epoch": m["best_epoch"], "stopped_epoch": m["completed_epochs"],
            "epochs_saved": m["epochs_saved"], "early_stopped": m["early_stopped"],
            "best_val_mse": m["best_val_mse"], "test_mse": m["test_mse"],
            "test_mae": m["test_mae"],
        })
        manifest.append({"run": key_text(key), "source_zip": item["archive"].name,
                         "data_hash": m["data_hash"],
                         "best_checkpoint_sha256": entry_digest(item, "checkpoint_best.pt")})
        if args.include_best or args.include_latest:
            folder = out / key_text(key)
            extract_entry(item, "metrics.json", folder / "metrics.json")
            if args.include_best:
                extract_entry(item, "checkpoint_best.pt", folder / "checkpoint_best.pt")
            if args.include_latest:
                extract_entry(item, "checkpoint_latest.pt", folder / "checkpoint_latest.pt")

    # Chỉ lưu bản consolidated vào repo; không chạm vào dữ liệu thô hay code huấn luyện.
    with (out / "metrics_merged.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    with (out / "merge_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=("run", "source_zip", "data_hash", "best_checkpoint_sha256"))
        w.writeheader()
        w.writerows(manifest)
    audit = {
        "target_runs": 225,
        "completed_runs": len(completed),
        "missing_runs": len(missing),
        "paused_runs": [key_text(k) for k in sorted(paused) if k not in completed],
        "missing_run_ids": [key_text(k) for k in missing],
        "source_archives": [p.name for p in archives],
        "archive_run_folder_counts": dict(source_counts),
        "runtime_signatures": signatures,
        "ready_for_final_analysis": len(completed) == 225,
    }
    (out / "merge_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã tạo:", out / "metrics_merged.csv")
    print("Đã tạo:", out / "merge_manifest.csv")
    print("Đã tạo:", out / "merge_audit.json")
    if len(completed) != 225:
        print("CHƯA ĐỦ 225 RUNS. Không sử dụng bản này làm báo cáo cuối cùng.")
    else:
        print("PASS: Đủ 225/225 runs để tổng hợp kết quả.")


if __name__ == "__main__":
    main()
