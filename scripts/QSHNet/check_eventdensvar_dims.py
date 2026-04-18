#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.configs import get_configs
from data.data_provider.data_factory import data_provider
from models.QSHNet import Model


DATASETS = [
    dict(
        dataset_name="HumanActivity",
        model_id="eventdensvar_main",
        collate_fn="collate_fn",
        seq_len=3000,
        pred_len=300,
        enc_in=12,
        dec_in=12,
        c_out=12,
        d_model=128,
        n_layers=3,
        n_heads=1,
        batch_size=2,
        dataset_root_path="storage/datasets/HumanActivity",
    ),
    dict(
        dataset_name="USHCN",
        model_id="eventdensvar_main",
        collate_fn="collate_fn",
        seq_len=150,
        pred_len=3,
        enc_in=5,
        dec_in=5,
        c_out=5,
        d_model=256,
        n_layers=1,
        n_heads=1,
        batch_size=2,
        dataset_root_path="storage/datasets/USHCN",
    ),
    dict(
        dataset_name="P12",
        model_id="eventdensvar_main",
        collate_fn="collate_fn",
        seq_len=36,
        pred_len=3,
        enc_in=36,
        dec_in=36,
        c_out=36,
        d_model=256,
        n_layers=2,
        n_heads=8,
        batch_size=2,
        dataset_root_path="storage/datasets/P12",
    ),
    dict(
        dataset_name="MIMIC_III",
        model_id="eventdensvar_main",
        collate_fn="collate_fn",
        seq_len=72,
        pred_len=3,
        enc_in=96,
        dec_in=96,
        c_out=96,
        d_model=256,
        n_layers=2,
        n_heads=4,
        batch_size=2,
        dataset_root_path="storage/datasets/MIMIC_III",
    ),
]


def build_configs(spec: dict) -> object:
    args = [
        "--model_name", "QSHNet",
        "--model_id", spec["model_id"],
        "--dataset_name", spec["dataset_name"],
        "--dataset_id", spec["dataset_name"],
        "--dataset_root_path", spec["dataset_root_path"],
        "--collate_fn", spec["collate_fn"],
        "--seq_len", str(spec["seq_len"]),
        "--pred_len", str(spec["pred_len"]),
        "--enc_in", str(spec["enc_in"]),
        "--dec_in", str(spec["dec_in"]),
        "--c_out", str(spec["c_out"]),
        "--d_model", str(spec["d_model"]),
        "--n_layers", str(spec["n_layers"]),
        "--n_heads", str(spec["n_heads"]),
        "--batch_size", str(spec["batch_size"]),
        "--num_workers", "0",
        "--itr", "1",
    ]
    return get_configs(args=args)


def main() -> None:
    print("=" * 80)
    print("QSHNet eventdensvar_main dimension smoke test")
    print("=" * 80)

    for spec in DATASETS:
        cfg = build_configs(spec)
        _, loader = data_provider(cfg, flag="train")
        batch = next(iter(loader))
        model = Model(cfg)
        output = model(
            **{
                "x": batch["x"],
                "x_mark": batch["x_mark"],
                "x_mask": batch["x_mask"],
                "y": batch["y"],
                "y_mark": batch["y_mark"],
                "y_mask": batch["y_mask"],
                "exp_stage": "test",
            }
        )

        assert batch["x"].shape[-1] == spec["enc_in"]
        assert batch["x_mask"].shape == batch["x"].shape
        assert batch["y"].shape[-1] == spec["c_out"]
        assert batch["y_mask"].shape == batch["y"].shape
        assert batch["x_mark"].shape[:2] == batch["x"].shape[:2]
        assert batch["y_mark"].shape[:2] == batch["y"].shape[:2]
        assert output["pred"].shape == output["true"].shape == batch["y"].shape

        print(f"\n[{spec['dataset_name']}]")
        print(f"  x       : {tuple(batch['x'].shape)}")
        print(f"  x_mask  : {tuple(batch['x_mask'].shape)}")
        print(f"  x_mark  : {tuple(batch['x_mark'].shape)}")
        print(f"  y       : {tuple(batch['y'].shape)}")
        print(f"  y_mask  : {tuple(batch['y_mask'].shape)}")
        print(f"  y_mark  : {tuple(batch['y_mark'].shape)}")
        print(f"  pred    : {tuple(output['pred'].shape)}")
        print("  status  : OK")

    print("\nALL_OK")


if __name__ == "__main__":
    main()
