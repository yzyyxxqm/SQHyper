#!/bin/bash
# =============================================================================
# QSH-Net Ablation Study (paper-grade)
#
# Configurations:
#   full     : full QSH-Net (qshnet_no_quat=0, qshnet_no_spike=0)   [optional]
#   no_quat  : disable Quaternion Refinement only
#   no_spike : disable Context-Aware Spike Selection only
#
# Datasets:
#   USHCN, HumanActivity, P12, MIMIC_III
#
# Hyper-parameters mirror scripts/QSHNet/run_all_itr5.sh exactly so that
# ablation rows are directly comparable to the main results table.
#
# Usage:
#   bash scripts/QSHNet/ablation.sh                              # no_quat + no_spike on all 4 datasets, itr=5
#   bash scripts/QSHNet/ablation.sh --configs no_quat            # filter configs
#   bash scripts/QSHNet/ablation.sh --datasets USHCN,P12         # filter datasets
#   bash scripts/QSHNet/ablation.sh --itr 3                      # change itr
#   bash scripts/QSHNet/ablation.sh --include-full               # also rerun full QSH-Net for sanity
# =============================================================================

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

. "$SCRIPT_DIR/../globals.sh"

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
CONFIGS="no_quat,no_spike"
DATASETS="USHCN,HumanActivity,P12,MIMIC_III"
ITR=5
INCLUDE_FULL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs)        CONFIGS="$2"; shift 2 ;;
        --datasets)       DATASETS="$2"; shift 2 ;;
        --itr)            ITR="$2"; shift 2 ;;
        --include-full)   INCLUDE_FULL=1; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# //;s/^#//'
            exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ "$INCLUDE_FULL" -eq 1 ]; then
    CONFIGS="full,$CONFIGS"
fi

LOG_DIR="$PROJECT_DIR/storage/logs/QSHNet_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "QSH-Net Ablation Study (itr=$ITR)"
echo "  configs : $CONFIGS"
echo "  datasets: $DATASETS"
echo "  log dir : $LOG_DIR"
echo "  start   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ---------------------------------------------------------------------------
# Per-config flag map
# ---------------------------------------------------------------------------
config_flags() {
    case "$1" in
        full)     echo "--qshnet_no_quat 0 --qshnet_no_spike 0" ;;
        no_quat)  echo "--qshnet_no_quat 1 --qshnet_no_spike 0" ;;
        no_spike) echo "--qshnet_no_quat 0 --qshnet_no_spike 1" ;;
        no_both)  echo "--qshnet_no_quat 1 --qshnet_no_spike 1" ;;
        *) echo "ERR: unknown config $1" >&2; exit 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Per-dataset hyper-parameters (must match run_all_itr5.sh)
# ---------------------------------------------------------------------------
dataset_hp() {
    # echoes: seq_len pred_len d_model n_layers n_heads batch_size
    case "$1" in
        USHCN)         echo "150  3   256 1 1 16" ;;
        HumanActivity) echo "3000 300 128 3 1 32" ;;
        P12)           echo "36   3   256 2 8 32" ;;
        MIMIC_III)     echo "72   3   256 2 4 32" ;;
        *) echo "ERR: unknown dataset $1" >&2; exit 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Run one (config, dataset) pair
# ---------------------------------------------------------------------------
run_one() {
    local cfg="$1"
    local ds="$2"

    read -r seq_len pred_len d_model n_layers n_heads batch_size <<< "$(dataset_hp "$ds")"
    local flags
    flags="$(config_flags "$cfg")"

    local dataset_subset_name=""
    local dataset_id="$ds"
    get_dataset_info "$ds" "$dataset_subset_name"

    local model_name="QSHNet"
    local model_id="QSHNet_${cfg}"   # distinct storage path per ablation config
    local log_file="$LOG_DIR/${ds}__${cfg}.log"

    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')] $ds × $cfg"
    echo "  flags=$flags"
    echo "  hp   = seq=$seq_len pred=$pred_len d=$d_model layers=$n_layers heads=$n_heads bs=$batch_size itr=$ITR"
    echo "  log  = $log_file"
    echo "------------------------------------------------------------"

    python main.py \
        --is_training 1 \
        --collate_fn "collate_fn" \
        --loss "MSE" \
        --d_model "$d_model" \
        --n_layers "$n_layers" \
        --n_heads "$n_heads" \
        --use_multi_gpu 0 \
        --dataset_root_path "$dataset_root_path" \
        --model_id "$model_id" \
        --model_name "$model_name" \
        --dataset_name "$ds" \
        --dataset_id "$dataset_id" \
        --features M \
        --seq_len "$seq_len" \
        --pred_len "$pred_len" \
        --enc_in "$n_variables" \
        --dec_in "$n_variables" \
        --c_out "$n_variables" \
        --train_epochs 300 \
        --patience 10 \
        --val_interval 1 \
        --itr "$ITR" \
        --batch_size "$batch_size" \
        --learning_rate 1e-3 \
        $flags \
        2>&1 | tee "$log_file"

    echo "[$(date '+%H:%M:%S')] done: $ds × $cfg"
}

# ---------------------------------------------------------------------------
# Sequential sweep (config × dataset)
# ---------------------------------------------------------------------------
IFS=',' read -ra CFG_ARR <<< "$CONFIGS"
IFS=',' read -ra DS_ARR  <<< "$DATASETS"

for cfg in "${CFG_ARR[@]}"; do
    for ds in "${DS_ARR[@]}"; do
        run_one "$cfg" "$ds"
    done
done

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Ablation complete. Aggregating..."
echo "============================================================"

python3 - "$LOG_DIR" "$CONFIGS" "$DATASETS" <<'PYTHON_SCRIPT'
import sys, re, json
from pathlib import Path
import numpy as np

log_dir = Path(sys.argv[1])
configs  = sys.argv[2].split(",")
datasets = sys.argv[3].split(",")

mse_pat = re.compile(r'"MSE":\s*([\d.eE+-]+)')
mae_pat = re.compile(r'"MAE":\s*([\d.eE+-]+)')

summary = {}
for cfg in configs:
    for ds in datasets:
        f = log_dir / f"{ds}__{cfg}.log"
        if not f.exists():
            print(f"[skip] {f} missing")
            continue
        text = f.read_text()
        mse = [float(x) for x in mse_pat.findall(text)]
        mae = [float(x) for x in mae_pat.findall(text)]
        if not mse:
            print(f"[warn] no MSE found in {f}")
            continue
        rec = {
            "mse_values": mse,
            "mae_values": mae,
            "mse_mean": float(np.mean(mse)),
            "mse_std":  float(np.std(mse, ddof=0)),
            "mae_mean": float(np.mean(mae)) if mae else None,
            "mae_std":  float(np.std(mae, ddof=0)) if mae else None,
        }
        summary.setdefault(ds, {})[cfg] = rec

print()
print(f"{'dataset':<14}{'config':<10}{'MSE (mean ± std)':<26}{'MAE (mean ± std)':<26}{'iters':>6}")
print("-" * 86)
for ds in datasets:
    for cfg in configs:
        rec = summary.get(ds, {}).get(cfg)
        if rec is None:
            print(f"{ds:<14}{cfg:<10}{'-':<26}{'-':<26}{'-':>6}")
            continue
        mse_str = f"{rec['mse_mean']:.4f} ± {rec['mse_std']:.4f}"
        mae_str = (f"{rec['mae_mean']:.4f} ± {rec['mae_std']:.4f}"
                   if rec['mae_mean'] is not None else "-")
        print(f"{ds:<14}{cfg:<10}{mse_str:<26}{mae_str:<26}{len(rec['mse_values']):>6}")

out = log_dir / "ablation_summary.json"
with out.open("w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {out}")
PYTHON_SCRIPT

echo ""
echo "============================================================"
echo "Done. End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log dir: $LOG_DIR"
echo "============================================================"
