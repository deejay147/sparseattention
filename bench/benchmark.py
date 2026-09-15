"""Benchmark (1.5): wall-clock + peak memory, dense vs sparse, forward only.

run from repo root on a GPU:  python bench/benchmark.py
writes results/benchmark.csv and plots/benchmark.png
"""
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")                  # no display in a script, just save png
import matplotlib.pyplot as plt
import pandas as pd
import torch

from sparse_attn.dense import dense_attention
from sparse_attn.masks import causal_mask, sliding_window_mask, bigbird_mask
from sparse_attn.sparse import sliding_window_attention

B, H, D = 1, 4, 64
DTYPE = torch.float32
W = 256                                # sliding window size
BB_BLOCK, BB_GLOBAL, BB_RANDOM = 64, 2, 2
NS = [512, 1024, 2048, 4096, 8192, 16384]   # 16384 past the brief, to see OOM
REPS, WARMUP = 5, 2


def measure(fn):
    """median time (ms) + peak memory above baseline (MB)"""
    for _ in range(WARMUP):             # first calls pick/cache cuda kernels, don't time them
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(REPS):
        torch.cuda.synchronize()        # gpu is async, wait before starting clock
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()        # and wait for it to actually finish
        times.append((time.perf_counter() - t0) * 1000)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()     # inputs + masks already here, excluded
    out = fn()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    del out
    return statistics.median(times), peak


@torch.inference_mode()                 # forward only, no autograd bookkeeping
def main():
    assert torch.cuda.is_available(), "needs a GPU (Colab: Runtime -> T4)"
    dev = "cuda"
    print(f"GPU:   {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}")
    print(f"config: B={B} H={H} d={D} dtype={DTYPE} w={W} "
          f"bigbird(block={BB_BLOCK}, global={BB_GLOBAL}, random={BB_RANDOM})\n")

    rows = []
    for N in NS:
        g = torch.Generator(device=dev).manual_seed(0)
        q, k, v = (torch.randn(B, H, N, D, device=dev, dtype=DTYPE, generator=g)
                   for _ in range(3))

        cm = causal_mask(N, device=dev)
        sm = sliding_window_mask(N, W, device=dev)
        bm = bigbird_mask(N, BB_BLOCK, BB_GLOBAL, BB_RANDOM, device=dev)

        methods = {
            "dense_causal":    lambda: dense_attention(q, k, v, cm),
            "sliding_masked":  lambda: dense_attention(q, k, v, sm),
            "bigbird_masked":  lambda: dense_attention(q, k, v, bm),
            "sliding_chunked": lambda: sliding_window_attention(q, k, v, W),
        }

        for name, fn in methods.items():
            try:
                t, mem = measure(fn)
                status = "ok"
            except torch.cuda.OutOfMemoryError:
                t, mem, status = float("nan"), float("nan"), "OOM"
            torch.cuda.empty_cache()         # clean up either way before next method
            rows.append(dict(method=name, N=N, time_ms=t, peak_mb=mem, status=status))
            print(f"N={N:<6d} {name:16s} {status:3s}  time={t:9.2f} ms  peak={mem:9.1f} MB")

        del q, k, v, cm, sm, bm
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "benchmark.csv", index=False)

    # relative numbers, dense_causal = 1.0 (brief wants relative)
    for col in ("time_ms", "peak_mb"):
        piv = df.pivot(index="N", columns="method", values=col)
        print(f"\n{col} relative to dense_causal:")
        print(piv.div(piv["dense_causal"], axis=0).round(3).to_string())

    # log-log plots: slope 2 = O(N^2), slope 1 = O(N)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, label in [(axes[0], "time_ms", "forward time (ms)"),
                           (axes[1], "peak_mb", "peak memory (MB)")]:
        for name, sub in df[df.status == "ok"].groupby("method"):
            ax.plot(sub.N, sub[col], marker="o", label=name)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("sequence length N")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(f"{torch.cuda.get_device_name(0)}, B={B} H={H} d={D} {str(DTYPE)[6:]}, w={W}")
    plt.tight_layout()
    (ROOT / "plots").mkdir(exist_ok=True)
    plt.savefig(ROOT / "plots" / "benchmark.png", dpi=120)
    print("\nsaved results/benchmark.csv, plots/benchmark.png")


if __name__ == "__main__":
    main()
