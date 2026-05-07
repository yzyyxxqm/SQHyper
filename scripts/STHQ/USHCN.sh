use_multi_gpu=0
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

. "$(dirname "$(readlink -f "$0")")/../globals.sh"

dataset_name=$(basename "$0" .sh)
dataset_subset_name=""
dataset_id=$dataset_name
get_dataset_info "$dataset_name" "$dataset_subset_name"

model_name="$(basename "$(dirname "$(readlink -f "$0")")")"
model_id=$model_name

# USHCN: V=5, seq_len=150
seq_len=150
for pred_len in 3; do
    $launch_command main.py \
    --is_training 1 \
    --collate_fn "collate_fn" \
    --loss "MSE_aux" \
    --d_model 256 \
    --n_layers 3 \
    --n_heads 1 \
    --use_multi_gpu $use_multi_gpu \
    --dataset_root_path $dataset_root_path \
    --model_id $model_id \
    --model_name $model_name \
    --dataset_name $dataset_name \
    --dataset_id $dataset_id \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $n_variables \
    --dec_in $n_variables \
    --c_out $n_variables \
    --train_epochs 300 \
    --patience 10 \
    --val_interval 1 \
    --itr 5 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --sthq_k_t 48 \
    --sthq_k_v 5 \
    --sthq_omega_min 0.02 \
    --sthq_omega_max 0.4 \
    --sthq_use_he_attn_from_layer 1
done
