#!/bin/bash
# QSH-Net Ablation Study
#
# Configurations:
#   A: Full QSH-Net (all components ON)
#   B: w/o Quaternion (noQB_noQH)
#   C: w/o Spiking (noSP)
#   D: w/o Causal Mask (noCM)
#   E: Pure HyperIMTS replica (noQB_noQH_noSP_noCM)
#   F: CausalMask only (noQB_noQH_noSP) — HyperIMTS + causal temporal masking
#
# Usage:
#   bash scripts/QSHNet/ablation.sh                 # all configs × 3 datasets, itr=1
#   bash scripts/QSHNet/ablation.sh E               # only config E, itr=1
#   bash scripts/QSHNet/ablation.sh E USHCN         # only config E on USHCN, itr=1
#   bash scripts/QSHNet/ablation.sh E "" 5          # config E, all datasets, itr=5
#   bash scripts/QSHNet/ablation.sh F "" 5          # config F, all datasets, itr=5

. "$(dirname "$(readlink -f "$0")")/../globals.sh"

model_name="QSHNet"
FILTER_CONFIG="${1:-}"
FILTER_DATASET="${2:-}"
ITR="${3:-1}"

run_one() {
    local ablation_name="$1"
    local model_id="$2"
    local dataset_name="$3"

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
    echo "  [$ablation_name] on $dataset_name  (itr=$ITR)"
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
        --itr $ITR \
        --batch_size $batch_size \
        --learning_rate 1e-3 \
        $extra_args
}

for ds in USHCN HumanActivity P12; do
    # Final model: QD + AS + QF (quaternion decoder + adaptive spike + quaternion fusion)
    run_one "QDASF" "QSHNet_noQB_noQH_noSP_noCM_noQV_noQE" "$ds"
    # Isolate QF contribution
    run_one "QF"    "QSHNet_noQB_noQH_noSP_noCM_noQV_noQE_noAS_noQD" "$ds"
    # Previous best: QD + AS (no QF)
    run_one "QDAS"  "QSHNet_noQB_noQH_noSP_noCM_noQV_noQE_noQF" "$ds"
    # AS only (previous best single component)
    run_one "AS"    "QSHNet_noQB_noQH_noSP_noCM_noQV_noQE_noQD_noQF" "$ds"
    # Pure HyperIMTS baseline
    run_one "E"     "QSHNet_noQB_noQH_noSP_noCM_noQV_noQE_noAS_noQD_noQF" "$ds"
done

echo ""
echo "=== Complete ==="
