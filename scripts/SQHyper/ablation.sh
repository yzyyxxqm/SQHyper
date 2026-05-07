#!/bin/bash
# SQHyper Ablation Study
# Runs 3 configurations: full, no_sgi, no_qmf
# on specified datasets

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
. "$SCRIPT_DIR/../globals.sh"

# Configuration
DATASETS="${1:-USHCN P12 HumanActivity MIMIC_III}"
ITR="${2:-5}"

echo "=== SQHyper Ablation Study ==="
echo "Datasets: $DATASETS"
echo "Iterations per config: $ITR"
echo ""

# Dataset-specific hyperparameters
get_hparams() {
    case "$1" in
        USHCN)      echo "256 1 1 150 3 16" ;;
        P12)        echo "64 1 1 150 3 16" ;;
        HumanActivity) echo "64 1 1 200 3 128" ;;
        MIMIC_III)  echo "256 2 4 72 3 32" ;;
        *)          echo "64 1 1 100 3 16" ;;
    esac
}

run_config() {
    local dataset=$1
    local config_name=$2
    local no_sgi=$3
    local no_qmf=$4

    get_dataset_info "$dataset" ""
    read d_model n_layers n_heads seq_len pred_len batch_size <<< $(get_hparams "$dataset")

    local model_id="SQHyper_${config_name}"
    echo "  [$config_name] d=$d_model layers=$n_layers heads=$n_heads bs=$batch_size"

    python main.py \
        --is_training 1 \
        --collate_fn "collate_fn" \
        --loss "MSE" \
        --d_model $d_model \
        --n_layers $n_layers \
        --n_heads $n_heads \
        --dataset_root_path $dataset_root_path \
        --model_id $model_id \
        --model_name SQHyper \
        --dataset_name $dataset \
        --dataset_id $dataset \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in $n_variables \
        --dec_in $n_variables \
        --c_out $n_variables \
        --train_epochs 300 \
        --patience 10 \
        --val_interval 1 \
        --itr $ITR \
        --batch_size $batch_size \
        --learning_rate 1e-3 \
        --sqhyper_no_sgi $no_sgi \
        --sqhyper_no_qmf $no_qmf \
        2>&1 | tail -5
}

for dataset in $DATASETS; do
    echo ""
    echo ">>> Dataset: $dataset ($(date))"
    echo "  --- full ---"
    run_config "$dataset" "full" 0 0
    echo "  --- no_sgi ---"
    run_config "$dataset" "no_sgi" 1 0
    echo "  --- no_qmf ---"
    run_config "$dataset" "no_qmf" 0 1
    echo ">>> Done: $dataset ($(date))"
done

echo ""
echo "=== Ablation study complete ==="
