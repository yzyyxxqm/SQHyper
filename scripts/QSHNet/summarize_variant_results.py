from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize latest metric.json results for a QSHNet variant.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names.")
    parser.add_argument("--checkpoints", default="storage/results")
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def find_latest_run(base: Path) -> Path | None:
    if not base.exists():
        return None
    metric_files = list(base.glob("iter*/eval_*/metric.json"))
    if metric_files:
        return base
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    for run in reversed(runs):
        if list(run.glob("iter*/eval_*/metric.json")):
            return run
    return None


def load_metrics(run_dir: Path) -> list[dict]:
    metrics = []
    for metric_file in sorted(run_dir.glob("iter*/eval_*/metric.json")):
        with open(metric_file, "r", encoding="utf-8") as f:
            metric = json.load(f)
        metrics.append(metric)
    return metrics


def main() -> None:
    args = parse_args()
    datasets = [ds.strip() for ds in args.datasets.split(",") if ds.strip()]
    checkpoints = Path(args.checkpoints)

    summary: dict[str, dict] = {}

    print("=" * 80)
    print(f"Summary for {args.model_name}/{args.model_id}")
    print("=" * 80)

    for dataset in datasets:
        base = checkpoints / dataset / dataset / args.model_name / args.model_id
        run_dir = find_latest_run(base)

        if run_dir is None:
            print(f"\n[{dataset}] missing results under {base}")
            summary[dataset] = {"status": "missing", "base": str(base)}
            continue

        metrics = load_metrics(run_dir)
        if not metrics:
            print(f"\n[{dataset}] no metric.json found under {run_dir}")
            summary[dataset] = {"status": "no_metrics", "run_dir": str(run_dir)}
            continue

        mse_values = [float(m["MSE"]) for m in metrics if "MSE" in m]
        mae_values = [float(m["MAE"]) for m in metrics if "MAE" in m]

        mse_mean = sum(mse_values) / len(mse_values)
        mse_std = statistics.stdev(mse_values) if len(mse_values) > 1 else 0.0
        mae_mean = sum(mae_values) / len(mae_values) if mae_values else None
        mae_std = statistics.stdev(mae_values) if len(mae_values) > 1 else 0.0

        print(f"\n[{dataset}] run_dir={run_dir}")
        print(f"  itr={len(mse_values)}")
        print(f"  MSE values: {', '.join(f'{v:.6f}' for v in mse_values)}")
        print(f"  MSE mean±std: {mse_mean:.6f} ± {mse_std:.6f}")
        if mae_values:
            print(f"  MAE mean±std: {mae_mean:.6f} ± {mae_std:.6f}")

        summary[dataset] = {
            "status": "ok",
            "run_dir": str(run_dir),
            "itr": len(mse_values),
            "mse_values": mse_values,
            "mse_mean": mse_mean,
            "mse_std": mse_std,
            "mae_values": mae_values,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
        }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved summary JSON to {output_path}")


if __name__ == "__main__":
    main()
