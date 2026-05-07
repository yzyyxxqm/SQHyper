#!/usr/bin/env bash
# =============================================================================
# QSH-Net Data-Scaling Ablation
#
# Tests the hypothesis: "QSH-Net's quaternion + spike modules act as inductive
# biases that pay off in the small-data regime, but become marginal once the
# training set is large." We retrain {full, no_quat, no_spike} on 4 fractions
# of the original training set and compare the ablation gap as a function of
# data scale.
#
# Configurations (subset of ablation.sh):
#   full     : full QSH-Net (qshnet_no_quat=0, qshnet_no_spike=0)
#   no_quat  : disable Quaternion Refinement only
#   no_spike : disable Context-Aware Spike Selection only
#
# Datasets: any of USHCN, HumanActivity, P12, MIMIC_III
#
# Usage:
#   bash scripts/QSHNet/data_scaling.sh                               # MIMIC_III × {full,no_quat,no_spike} × {0.10,0.25}
#   bash scripts/QSHNet/data_scaling.sh --datasets USHCN              # different dataset
#   bash scripts/QSHNet/data_scaling.sh --configs no_spike            # filter configs
#   bash scripts/QSHNet/data_scaling.sh --fractions 0.10,0.25,0.50    # filter fractions
#   bash scripts/QSHNet/data_scaling.sh --itr 3                       # change itr
#   bash scripts/QSHNet/data_scaling.sh --frac-seed 42                # change subsample seed
#
# Notes:
#   * fraction=1.0 is intentionally NOT included by default — those numbers
#     come from the existing main results / ablation.sh runs (model_id has no
#     `_frac` suffix there). Adding `--fractions ...,1.0` here will create a
#     separate `_frac100` directory which is useful only as a sanity check.
#   * model_id = QSHNet_${cfg}_frac${pct}  (e.g. QSHNet_no_spike_frac10)
#     so storage paths never collide with ablation.sh or run_all_itr5.sh.
# =============================================================================

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

. "$SCRIPT_DIR/../globals.sh"

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
CONFIGS="full,no_quat,no_spike"
DATASETS="MIMIC_III"
FRACTIONS="0.10,0.25"
ITR=5
FRAC_SEED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs)        CONFIGS="$2"; shift 2 ;;
        --datasets)       DATASETS="$2"; shift 2 ;;
        --fractions)      FRACTIONS="$2"; shift 2 ;;
        --itr)            ITR="$2"; shift 2 ;;
        --frac-seed)      FRAC_SEED="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# //;s/^#//'
            exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

LOG_DIR="$PROJECT_DIR/storage/logs/QSHNet_data_scaling_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "QSH-Net Data-Scaling Ablation (itr=$ITR, frac_seed=$FRAC_SEED)"
echo "  configs   : $CONFIGS"
echo "  datasets  : $DATASETS"
echo "  fractions : $FRACTIONS"
echo "  log dir   : $LOG_DIR"
echo "  start     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ---------------------------------------------------------------------------
# Per-config flag map (mirrors ablation.sh exactly)
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
# Per-dataset hyper-parameters (must match run_all_itr5.sh / ablation.sh)
# ---------------------------------------------------------------------------
dataset_hp() {
    case "$1" in
        USHCN)         echo "150  3   256 1 1 16" ;;
        HumanActivity) echo "3000 300 128 3 1 32" ;;
        P12)           echo "36   3   256 2 8 32" ;;
        MIMIC_III)     echo "72   3   256 2 4 32" ;;
        *) echo "ERR: unknown dataset $1" >&2; exit 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# fraction → integer percent (for filesystem-safe model_id suffix)
# ---------------------------------------------------------------------------
frac_pct() {
    # 0.10 -> 10, 0.25 -> 25, 0.5 -> 50, 1.0 -> 100
    python3 -c "import sys; v=float(sys.argv[1]); print(int(round(v*100)))" "$1"
}

# ---------------------------------------------------------------------------
# Run one (config, dataset, fraction) cell
# ---------------------------------------------------------------------------
run_one() {
    local cfg="$1"
    local ds="$2"
    local frac="$3"

    read -r seq_len pred_len d_model n_layers n_heads batch_size <<< "$(dataset_hp "$ds")"
    local flags
    flags="$(config_flags "$cfg")"

    local pct
    pct="$(frac_pct "$frac")"

    local dataset_subset_name=""
    local dataset_id="${ds}_frac${pct}"
    get_dataset_info "$ds" "$dataset_subset_name"

    local model_name="QSHNet"
    local model_id="QSHNet_${cfg}_frac${pct}"
    local log_file="$LOG_DIR/${ds}__${cfg}__frac${pct}.log"

    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')] $ds × $cfg × frac=$frac (pct=$pct)"
    echo "  flags=$flags --train_fraction $frac --train_fraction_seed $FRAC_SEED"
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
        --train_fraction "$frac" \
        --train_fraction_seed "$FRAC_SEED" \
        $flags \
        2>&1 | tee "$log_file"

    echo "[$(date '+%H:%M:%S')] done: $ds × $cfg × frac=$frac"
}

# ---------------------------------------------------------------------------
# Sweep order: fraction (outer, smaller first → faster feedback)
#              config   (middle)
#              dataset  (inner)
# ---------------------------------------------------------------------------
IFS=',' read -ra CFG_ARR  <<< "$CONFIGS"
IFS=',' read -ra DS_ARR   <<< "$DATASETS"
IFS=',' read -ra FRAC_ARR <<< "$FRACTIONS"

for frac in "${FRAC_ARR[@]}"; do
    for cfg in "${CFG_ARR[@]}"; do
        for ds in "${DS_ARR[@]}"; do
            run_one "$cfg" "$ds" "$frac"
        done
    done
done

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Data-scaling sweep complete. Aggregating..."
echo "============================================================"

python3 - "$LOG_DIR" "$CONFIGS" "$DATASETS" "$FRACTIONS" <<'PYTHON_SCRIPT'
import sys, re, json
from pathlib import Path
import numpy as np

log_dir   = Path(sys.argv[1])
configs   = sys.argv[2].split(",")
datasets  = sys.argv[3].split(",")
fractions = [float(x) for x in sys.argv[4].split(",")]

mse_pat = re.compile(r'"MSE":\s*([\d.eE+-]+)')
mae_pat = re.compile(r'"MAE":\s*([\d.eE+-]+)')

summary = {}
for frac in fractions:
    pct = int(round(frac * 100))
    for cfg in configs:
        for ds in datasets:
            f = log_dir / f"{ds}__{cfg}__frac{pct}.log"
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
                "n_iters":  len(mse),
            }
            summary.setdefault(ds, {}).setdefault(frac, {})[cfg] = rec

print()
hdr = f"{'dataset':<14}{'frac':<6}{'config':<10}{'MSE (mean ± std)':<26}{'MAE (mean ± std)':<26}{'iters':>6}"
print(hdr)
print("-" * len(hdr))
for ds in datasets:
    for frac in fractions:
        for cfg in configs:
            rec = summary.get(ds, {}).get(frac, {}).get(cfg)
            if rec is None:
                print(f"{ds:<14}{frac:<6.2f}{cfg:<10}{'-':<26}{'-':<26}{'-':>6}")
                continue
            mse_str = f"{rec['mse_mean']:.4f} ± {rec['mse_std']:.4f}"
            mae_str = (f"{rec['mae_mean']:.4f} ± {rec['mae_std']:.4f}"
                       if rec['mae_mean'] is not None else "-")
            print(f"{ds:<14}{frac:<6.2f}{cfg:<10}{mse_str:<26}{mae_str:<26}{rec['n_iters']:>6}")

out = log_dir / "data_scaling_summary.json"
with out.open("w") as f:
    json.dump({str(k1): {str(k2): v2 for k2, v2 in v1.items()}
               for k1, v1 in summary.items()}, f, indent=2)
print(f"\nSaved: {out}")
PYTHON_SCRIPT

echo ""
echo "============================================================"
echo "Done. End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log dir: $LOG_DIR"
echo "============================================================"
