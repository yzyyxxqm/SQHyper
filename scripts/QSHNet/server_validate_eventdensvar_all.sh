#!/bin/bash
# =============================================================================
# QSH-Net eventdensvar_main 全数据集服务器验证脚本
# 默认按当前约定运行：
# - USHCN: itr=10
# - HumanActivity: itr=5
# - P12 / MIMIC_III: itr=5
# - 默认并行启动 HumanActivity / USHCN / P12 / MIMIC_III
# 用法:
#   bash scripts/QSHNet/server_validate_eventdensvar_all.sh
# 可选环境变量:
#   USE_MULTI_GPU=1
#   ITR_USHCN=5
#   ITR_HUMAN=5
#   ITR_P12=5
#   ITR_MIMIC_III=5
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

. "$SCRIPT_DIR/../globals.sh"

USE_MULTI_GPU="${USE_MULTI_GPU:-0}"
if [ "$USE_MULTI_GPU" -eq 0 ]; then
    LAUNCH_COMMAND="python"
else
    LAUNCH_COMMAND="accelerate launch"
fi

MODEL_NAME="QSHNet"
MODEL_ID="eventdensvar_main"

ITR_USHCN="${ITR_USHCN:-10}"
ITR_HUMAN="${ITR_HUMAN:-5}"
ITR_P12="${ITR_P12:-5}"
ITR_MIMIC_III="${ITR_MIMIC_III:-5}"

LOG_DIR="$PROJECT_DIR/storage/logs/${MODEL_ID}_server_validate_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "QSH-Net 服务器验证: $MODEL_ID"
echo "项目目录: $PROJECT_DIR"
echo "日志目录: $LOG_DIR"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

run_dataset() {
    local dataset_name="$1"
    local seq_len="$2"
    local pred_len="$3"
    local d_model="$4"
    local n_layers="$5"
    local n_heads="$6"
    local batch_size="$7"
    local itr="$8"

    local dataset_subset_name=""
    local dataset_id="$dataset_name"
    get_dataset_info "$dataset_name" "$dataset_subset_name"

    local log_file="$LOG_DIR/${dataset_name}.log"

    echo
    echo "------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')] 开始: $dataset_name"
    echo "model_id=$MODEL_ID seq_len=$seq_len pred_len=$pred_len d_model=$d_model n_layers=$n_layers n_heads=$n_heads batch_size=$batch_size itr=$itr"
    echo "日志: $log_file"
    echo "------------------------------------------------------------"

    PYTHONUNBUFFERED=1 $LAUNCH_COMMAND main.py \
        --is_training 1 \
        --collate_fn "collate_fn" \
        --loss "MSE" \
        --d_model "$d_model" \
        --n_layers "$n_layers" \
        --n_heads "$n_heads" \
        --use_multi_gpu "$USE_MULTI_GPU" \
        --dataset_root_path "$dataset_root_path" \
        --model_id "$MODEL_ID" \
        --model_name "$MODEL_NAME" \
        --dataset_name "$dataset_name" \
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
        --itr "$itr" \
        --batch_size "$batch_size" \
        --learning_rate 1e-3 \
        2>&1 | tee "$log_file"

    echo "[$(date '+%H:%M:%S')] 完成: $dataset_name"
}

PIDS=()
DATASETS=()

run_dataset "HumanActivity" 3000 300 128 3 1 32 "$ITR_HUMAN" &
PIDS+=($!)
DATASETS+=("HumanActivity")

run_dataset "USHCN" 150 3 256 1 1 16 "$ITR_USHCN" &
PIDS+=($!)
DATASETS+=("USHCN")

run_dataset "P12" 36 3 256 2 8 32 "$ITR_P12" &
PIDS+=($!)
DATASETS+=("P12")

run_dataset "MIMIC_III" 72 3 256 2 4 32 "$ITR_MIMIC_III" &
PIDS+=($!)
DATASETS+=("MIMIC_III")

FAILURES=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    dataset="${DATASETS[$i]}"
    if wait "$pid"; then
        echo "[OK] $dataset"
    else
        echo "[FAIL] $dataset"
        FAILURES=$((FAILURES + 1))
    fi
done

if [ "$FAILURES" -ne 0 ]; then
    echo "共有 $FAILURES 个并行任务失败，仍继续执行结果汇总。"
fi

echo
echo "============================================================"
echo "训练结束，开始汇总结果"
echo "============================================================"

python scripts/QSHNet/summarize_variant_results.py \
    --model_name "$MODEL_NAME" \
    --model_id "$MODEL_ID" \
    --datasets "HumanActivity,USHCN,P12,MIMIC_III" \
    --output_json "$LOG_DIR/summary.json" \
    2>&1 | tee "$LOG_DIR/summary.log"

echo
echo "完成。日志目录: $LOG_DIR"
