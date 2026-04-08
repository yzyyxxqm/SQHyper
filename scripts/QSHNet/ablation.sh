#!/bin/bash
# QSH-Net Ablation Study
#
# 5 configurations:
#   A: Full QSH-Net (all components ON)
#   B: w/o Quaternion (noQB_noQH)
#   C: w/o Spiking (noSP)
#   D: w/o Causal Mask (noCM)
#   E: Pure HyperIMTS replica (noQB_noQH_noSP_noCM)
#
# Usage:
#   bash scripts/QSHNet/ablation.sh           # run all 5 configs × 3 datasets
#   bash scripts/QSHNet/ablation.sh E          # run only config E
#   bash scripts/QSHNet/ablation.sh E USHCN    # run only config E on USHCN

. "$(dirname "$(readlink -f "$0")")/../globals.sh"

model_name="QSHNet"
FILTER_CONFIG="${1:-}"
FILTER_DATASET="${2:-}"

run_one() {
    local ablation_name="$1"
    local model_id="$2"
    local dataset_name="$3"

    # Skip if filter is set and doesn't match
    if [ -n "$FILTER_CONFIG" ] && [ "$FILTER_CONFIG" != "$ablation_name" ]; then return; fi
    if [ -n "$FILTER_DATASET" ] && [ "$FILTER_DATASET" != "$dataset_name" ]; then return; fi

    get_dataset_info "$dataset_name" ""

    local extra_args=""
    local d_model=128
    local n_layers=2
    local n_heads=4
    local batch_size=32
    local seq_len=96
    local pred_len=3

    case $dataset_name in
        P12)
            d_model=256; n_layers=1; n_heads=8; seq_len=36; pred_len=3
            extra_args="--collate_fn collate_fn"
            ;;
        HumanActivity)
            d_model=128; n_layers=3; n_heads=1; seq_len=3000; pred_len=300
            extra_args="--collate_fn collate_fn"
            ;;
        USHCN)
            d_model=256; n_layers=1; n_heads=1; seq_len=150; pred_len=3; batch_size=16
            extra_args="--collate_fn collate_fn"
            ;;
    esac

    echo ""
    echo "============================================"
    echo "  [$ablation_name] on $dataset_name"
    echo "  model_id=$model_id"
    echo "============================================"
    python main.py \
        --is_training 1 \
        --loss "MSE" \
        --d_model $d_model \
        --n_layers $n_layers \
        --n_heads $n_heads \
        --use_multi_gpu 0 \
        --dataset_root_path $dataset_root_path \
        --model_id "$model_id" \
        --model_name $model_name \
        --dataset_name $dataset_name \
        --dataset_id $dataset_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in $n_variables \
        --dec_in $n_variables \
        --c_out $n_variables \
        --train_epochs 300 \
        --patience 10 \
        --val_interval 1 \
        --itr 1 \
        --batch_size $batch_size \
        --learning_rate 1e-3 \
        $extra_args
}

# Config A: Full QSH-Net
for ds in USHCN HumanActivity P12; do
    run_one "A" "QSHNet" "$ds"
done

# Config B: w/o Quaternion
for ds in USHCN HumanActivity P12; do
    run_one "B" "QSHNet_noQB_noQH" "$ds"
done

# Config C: w/o Spiking
for ds in USHCN HumanActivity P12; do
    run_one "C" "QSHNet_noSP" "$ds"
done

# Config D: w/o Causal Mask
for ds in USHCN HumanActivity P12; do
    run_one "D" "QSHNet_noCM" "$ds"
done

# Config E: Pure HyperIMTS replica
for ds in USHCN HumanActivity P12; do
    run_one "E" "QSHNet_noQB_noQH_noSP_noCM" "$ds"
done

echo ""
echo "=== Ablation study complete ==="
