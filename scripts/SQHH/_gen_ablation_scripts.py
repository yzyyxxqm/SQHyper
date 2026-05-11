#!/usr/bin/env python3
"""Generate USHCN/P12/MIMIC_III/HumanActivity ablation triplets for SQHH."""
import os
from pathlib import Path

HERE = Path(__file__).parent

# (dataset, seq_len, pred_len, d_model, n_layers, n_heads, batch_size,
#  k_a, k_e, spike_floor)
CONFIGS = {
    "USHCN":         (150,   3, 256, 1, 1, 16, 16, 24, 0.1),
    "P12":           (36,    3, 256, 1, 8, 32, 16, 32, 0.2),
    "MIMIC_III":     (72,    3, 256, 2, 4, 32, 24, 48, 0.2),
    "HumanActivity": (3000, 300, 128, 3, 1, 32, 12, 24, 0.15),
}

ABLATIONS = {
    "no_sri":  ["--sqhh_no_sri 1"],
    "no_sqc":  ["--sqhh_no_sqc 1"],
    "no_both": ["--sqhh_no_sri 1", "--sqhh_no_sqc 1"],
}

TEMPLATE = """use_multi_gpu=0
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

. "$(dirname "$(readlink -f "$0")")/../globals.sh"

dataset_name="{ds}"
dataset_subset_name=""
dataset_id=$dataset_name
get_dataset_info "$dataset_name" "$dataset_subset_name"

model_name="SQHH"
model_id="SQHH_{abl}"

# SQHH ablation: {abl}
seq_len={seq_len}
for pred_len in {pred_len}; do
    $launch_command main.py \\
    --is_training 1 \\
    --collate_fn "collate_fn" \\
    --loss "MSE" \\
    --d_model {d_model} \\
    --n_layers {n_layers} \\
    --n_heads {n_heads} \\
    --use_multi_gpu $use_multi_gpu \\
    --dataset_root_path $dataset_root_path \\
    --model_id $model_id \\
    --model_name $model_name \\
    --dataset_name $dataset_name \\
    --dataset_id $dataset_id \\
    --features M \\
    --seq_len $seq_len \\
    --pred_len $pred_len \\
    --enc_in $n_variables \\
    --dec_in $n_variables \\
    --c_out $n_variables \\
    --train_epochs 300 \\
    --patience 10 \\
    --val_interval 1 \\
    --itr 1 \\
    --batch_size {batch_size} \\
    --learning_rate 1e-3 \\
    --sqhh_k_a {k_a} \\
    --sqhh_k_e {k_e} \\
    --sqhh_spike_floor {spike_floor} \\
{flag_lines}
    --sqhh_diag_interval 200
done
"""

for ds, (seq, pl, dm, nl, nh, bs, ka, ke, sf) in CONFIGS.items():
    for abl, flags in ABLATIONS.items():
        flag_lines = "\n".join(f"    {f} \\" for f in flags)
        content = TEMPLATE.format(
            ds=ds, abl=abl, seq_len=seq, pred_len=pl,
            d_model=dm, n_layers=nl, n_heads=nh, batch_size=bs,
            k_a=ka, k_e=ke, spike_floor=sf,
            flag_lines=flag_lines,
        )
        out = HERE / f"{ds}_{abl}.sh"
        out.write_text(content)
        print(f"wrote {out.name}")
print("done.")
