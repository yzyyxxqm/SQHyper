"""GPU per-batch timing benchmark for SQHH v2 optimization rollout.

Compares fp32 / bf16 / bf16+compile on a P12-sized synthetic batch.
Measures forward and forward+backward separately; runs warmup then averages.

Usage:
    python -m tests.sqhh_v2_speed_bench --dataset p12
    python -m tests.sqhh_v2_speed_bench --dataset mimic
"""
import argparse
import contextlib
import time

import torch

from tests.sqhh_v2_bit_exact_smoke import make_cfg
from models.SQHH import Model


PRESETS = {
    # name: (d_model, enc_in, n_layers, seq_len, pred_len, batch_size)
    "p12":    (256, 36, 2, 36, 3, 32),
    "mimic":  (256, 96, 2, 36, 3, 32),
    "ushcn":  (256, 5, 2, 150, 3, 16),
    "ha":     (128, 12, 3, 3000, 100, 32),
    "small":  (32, 8, 2, 24, 6, 8),
}


def make_batch(cfg, density=0.4, device="cuda"):
    B = cfg.batch_size
    L = cfg.seq_len
    E = cfg.enc_in
    Lp = cfg.pred_len
    x = torch.randn(B, L, E, device=device)
    x_mask = (torch.rand_like(x) > (1 - density)).float()
    y = torch.randn(B, Lp, E, device=device)
    y_mask = (torch.rand_like(y) > 0.5).float()
    return dict(x=x, x_mask=x_mask, y=y, y_mask=y_mask)


def run_one(model, batch, mode, n_warmup=3, n_iter=10):
    """mode: 'fp32', 'bf16', 'fp16'"""
    if mode == "bf16":
        ctx_factory = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    elif mode == "fp16":
        ctx_factory = lambda: torch.autocast("cuda", dtype=torch.float16)
    else:
        ctx_factory = lambda: contextlib.nullcontext()

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Warmup
    for _ in range(n_warmup):
        optim.zero_grad()
        with ctx_factory():
            out = model(**batch, exp_stage="train")
            loss = (out["pred"] * out["mask"].float()).pow(2).sum()
        loss.backward()
        optim.step()
    torch.cuda.synchronize()

    # Forward only
    t0 = time.perf_counter()
    for _ in range(n_iter):
        with torch.no_grad():
            with ctx_factory():
                out = model(**batch, exp_stage="val")
                loss = (out["pred"] * out["mask"].float()).pow(2).sum()
    torch.cuda.synchronize()
    fwd_time = (time.perf_counter() - t0) / n_iter

    # Forward + backward + step
    t0 = time.perf_counter()
    for _ in range(n_iter):
        optim.zero_grad()
        with ctx_factory():
            out = model(**batch, exp_stage="train")
            loss = (out["pred"] * out["mask"].float()).pow(2).sum()
        loss.backward()
        optim.step()
    torch.cuda.synchronize()
    full_time = (time.perf_counter() - t0) / n_iter

    return fwd_time, full_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="p12", choices=list(PRESETS))
    ap.add_argument("--no_compile", action="store_true",
                    help="skip torch.compile benchmark")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; aborting.")
        return

    d_model, enc_in, n_layers, seq_len, pred_len, batch_size = PRESETS[args.dataset]
    cfg = make_cfg(d_model=d_model, enc_in=enc_in, n_layers=n_layers,
                    seq_len=seq_len, pred_len=pred_len, batch_size=batch_size)

    print(f"Dataset preset: {args.dataset}")
    print(f"  d_model={d_model} enc_in={enc_in} n_layers={n_layers} "
          f"seq_len={seq_len} pred_len={pred_len} batch_size={batch_size}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print()

    print(f"{'Mode':<22s} {'fwd ms':>10s} {'full ms':>10s} {'fwd x':>8s} {'full x':>8s}")
    print("-" * 64)

    results = {}
    for mode in ["fp32", "bf16"]:
        torch.manual_seed(0)
        model = Model(cfg).cuda().train()
        batch = make_batch(cfg, density=0.4)
        fwd, full = run_one(model, batch, mode)
        results[mode] = (fwd, full)
        del model
        torch.cuda.empty_cache()
        base_fwd, base_full = results.get("fp32", (fwd, full))
        print(f"{mode:<22s} {fwd*1000:>9.2f}  {full*1000:>9.2f}  "
              f"{base_fwd/fwd:>7.2f}x {base_full/full:>7.2f}x")

    if not args.no_compile:
        torch.manual_seed(0)
        model = Model(cfg).cuda().train()
        try:
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, dynamic=True, mode="default")
        except Exception as e:
            print(f"compile failed: {e}")
        batch = make_batch(cfg, density=0.4)
        try:
            fwd, full = run_one(model, batch, "bf16", n_warmup=5)
            base_fwd, base_full = results["fp32"]
            print(f"{'bf16+compile':<22s} {fwd*1000:>9.2f}  {full*1000:>9.2f}  "
                  f"{base_fwd/fwd:>7.2f}x {base_full/full:>7.2f}x")
        except Exception as e:
            print(f"bf16+compile run failed: {e}")
        del model
        torch.cuda.empty_cache()

    print("\nNote: full = forward + backward + optimizer step. "
          "x = speedup vs fp32.")


if __name__ == "__main__":
    main()
