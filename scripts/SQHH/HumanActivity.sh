use_multi_gpu=0
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

. "$(dirname "$(readlink -f "$0")")/../globals.sh"

dataset_name="HumanActivity"
dataset_subset_name=""
dataset_id=$dataset_name
get_dataset_info "$dataset_name" "$dataset_subset_name"

model_name="SQHH"
model_id=$model_name

# Training config aligned 1:1 with scripts/HyperIMTS/HumanActivity.sh.
# CRITICAL: HyperIMTS uses pred_len=300 (NOT 3). Previous SQHH runs used
# pred_len=3 which gave artificially low MSE 0.0207 — that 'win' was on a
# different (much easier) task.
seq_len=3000
for pred_len in 300; do
    $launch_command main.py \
    --is_training 1 \
    --collate_fn "collate_fn" \
    --loss "MSE" \
    --d_model 128 \
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
    --batch_size 32 \
    --learning_rate 1e-3 \
    --sqhh_k_a 12 \
    --sqhh_k_e 24 \
    --sqhh_spike_floor 0.15 \
    --sqhh_diag_interval 200
done
