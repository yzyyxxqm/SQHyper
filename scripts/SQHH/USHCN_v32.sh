use_multi_gpu=0
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

. "$(dirname "$(readlink -f "$0")")/../globals.sh"

dataset_name="USHCN"
dataset_subset_name=""
dataset_id=$dataset_name
get_dataset_info "$dataset_name" "$dataset_subset_name"

model_name="SQHH"
model_id="SQHH"

# v3.2: keep best config from v3 (d_model=256 bf16) but slower lr + longer
# patience. Hypothesis: model has right capacity for USHCN but needs
# gentler convergence to escape bad seeds.
seq_len=150
for pred_len in 3; do
    $launch_command main.py \
    --is_training 1 \
    --collate_fn "collate_fn" \
    --loss "MSE" \
    --d_model 256 \
    --n_layers 2 \
    --n_heads 2 \
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
    --patience 40 \
    --val_interval 1 \
    --itr 5 \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --use_amp 1 \
    --use_compile 0 \
    --sqhh_k_a 16 \
    --sqhh_k_e 24 \
    --sqhh_spike_floor 0.1 \
    --sqhh_diag_interval 200
done
