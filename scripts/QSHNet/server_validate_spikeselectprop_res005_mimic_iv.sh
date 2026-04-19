#!/bin/bash
# =============================================================================
# QSH-Net spikeselectprop_res005_itr10 MIMIC-IV 服务器验证脚本
# 默认只跑 MIMIC_IV，避免和全数据集并行脚本混在一起。
#
# 用法：
#   bash scripts/QSHNet/server_validate_spikeselectprop_res005_mimic_iv.sh
#
# 常用覆盖：
#   USE_MULTI_GPU=1 bash scripts/QSHNet/server_validate_spikeselectprop_res005_mimic_iv.sh
#   ITR_MIMIC_IV=1 bash scripts/QSHNet/server_validate_spikeselectprop_res005_mimic_iv.sh
#   BATCH_SIZE_MIMIC_IV=16 bash scripts/QSHNet/server_validate_spikeselectprop_res005_mimic_iv.sh
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
MODEL_ID="spikeselectprop_res005_itr10"
DATASET_NAME="MIMIC_IV"
DATASET_SUBSET_NAME=""
DATASET_ID="$DATASET_NAME"

ITR_MIMIC_IV="${ITR_MIMIC_IV:-5}"
BATCH_SIZE_MIMIC_IV="${BATCH_SIZE_MIMIC_IV:-32}"
USE_NUM_WORKERS="${USE_NUM_WORKERS:-0}"

SEQ_LEN=2160
PRED_LEN=3
D_MODEL=128
N_LAYERS=4
N_HEADS=8

get_dataset_info "$DATASET_NAME" "$DATASET_SUBSET_NAME"

LOG_DIR="$PROJECT_DIR/storage/logs/${MODEL_ID}_mimic_iv_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/MIMIC_IV.log"

cat <<EOF
============================================================
QSH-Net MIMIC-IV 验证: $MODEL_ID
项目目录: $PROJECT_DIR
日志目录: $LOG_DIR
开始时间: $(date '+%Y-%m-%d %H:%M:%S')
参数: seq_len=$SEQ_LEN pred_len=$PRED_LEN d_model=$D_MODEL n_layers=$N_LAYERS n_heads=$N_HEADS batch_size=$BATCH_SIZE_MIMIC_IV itr=$ITR_MIMIC_IV n_variables=$n_variables
============================================================
EOF

PYTHONUNBUFFERED=1 $LAUNCH_COMMAND main.py \
    --is_training 1 \
    --collate_fn "collate_fn" \
    --loss "MSE" \
    --d_model "$D_MODEL" \
    --n_layers "$N_LAYERS" \
    --n_heads "$N_HEADS" \
    --use_multi_gpu "$USE_MULTI_GPU" \
    --dataset_root_path "$dataset_root_path" \
    --model_id "$MODEL_ID" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --dataset_id "$DATASET_ID" \
    --features M \
    --seq_len "$SEQ_LEN" \
    --pred_len "$PRED_LEN" \
    --enc_in "$n_variables" \
    --dec_in "$n_variables" \
    --c_out "$n_variables" \
    --train_epochs 300 \
    --patience 10 \
    --val_interval 1 \
    --itr "$ITR_MIMIC_IV" \
    --batch_size "$BATCH_SIZE_MIMIC_IV" \
    --num_workers "$USE_NUM_WORKERS" \
    --learning_rate 1e-3 \
    2>&1 | tee "$LOG_FILE"

python scripts/QSHNet/summarize_variant_results.py \
    --model_name "$MODEL_NAME" \
    --model_id "$MODEL_ID" \
    --datasets "MIMIC_IV" \
    --output_json "$LOG_DIR/summary.json" \
    2>&1 | tee "$LOG_DIR/summary.log"

echo
echo "完成。日志目录: $LOG_DIR"
