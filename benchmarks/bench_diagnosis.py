"""
Benchmark: Hat matrix computation time and memory at various dataset sizes.

Demonstrates the O(n^2) growth that motivates LITE_THRESHOLD=10000.

Usage:
    python benchmarks/bench_diagnosis.py
"""

import gc
import time

import numpy as np
import torch

SIZES = [1000, 2000, 5000, 7500, 10000]
K = 3  # number of features


def simulate_hat_matrix(n, k):
    """Simulate the Hat matrix computation from DIAGNOSIS.__init__."""
    x_data = torch.randn(n, k)
    weight = torch.randn(n, k)  # same shape as x_data

    hat_com = torch.mm(
        torch.linalg.inv(torch.mm(x_data.t(), x_data)),
        x_data.t()
    )
    ols_hat = torch.mm(x_data, hat_com)

    x_data_tile = x_data.repeat(n, 1).view(n, n, -1)
    x_data_tile_t = x_data_tile.transpose(1, 2)
    gtweight_3d = torch.diag_embed(weight)

    hatS_temp = torch.matmul(
        gtweight_3d,
        torch.matmul(torch.inverse(torch.matmul(x_data_tile_t, x_data_tile)),
                      x_data_tile_t)
    )
    hatS = torch.matmul(x_data.view(-1, 1, x_data.size(1)), hatS_temp)
    hatS = hatS.view(-1, n)
    S = torch.trace(hatS)
    return hatS


def estimate_memory_mb(n, k):
    """Estimate theoretical memory for Hat matrix computation.

    Key tensors:
    - x_data_tile: (n, n, k) float32 = n^2 * k * 4 bytes
    - gtweight_3d: (n, k, k) float32 = n * k^2 * 4 bytes
    - hatS_temp:   (n, k, n) float32 = n^2 * k * 4 bytes
    - hatS:        (n, n)    float32 = n^2 * 4 bytes
    Total dominant: ~(2*k + 1) * n^2 * 4 bytes
    """
    bytes_total = (2 * k + 1) * n * n * 4 + n * k * k * 4
    return bytes_total / 1024 / 1024


def main():
    print("=" * 60)
    print("Hat Matrix Computation Benchmark")
    print(f"Features k={K}")
    print("=" * 60)

    print(f"\n| {'n':>6} | {'time (s)':>10} | {'est. mem (MB)':>13} | {'n^2 ratio':>10} |")
    print(f"|{'-'*8}|{'-'*12}|{'-'*15}|{'-'*12}|")

    base_time = None
    for n in SIZES:
        mem_mb = estimate_memory_mb(n, K)

        gc.collect()
        t0 = time.perf_counter()
        simulate_hat_matrix(n, K)
        elapsed = time.perf_counter() - t0

        if base_time is None:
            base_time = elapsed
            ratio_str = "1.0x"
        else:
            expected_ratio = (n / SIZES[0]) ** 2
            actual_ratio = elapsed / base_time
            ratio_str = f"{actual_ratio:.1f}x (expect {expected_ratio:.0f}x)"

        print(f"| {n:>6} | {elapsed:>10.3f} | {mem_mb:>11.0f}  | {ratio_str:>10} |")

    print(f"\nTheoretical memory at various scales:")
    for n in [10000, 20000, 50000, 100000]:
        mem = estimate_memory_mb(n, K)
        unit = "MB" if mem < 1024 else "GB"
        val = mem if mem < 1024 else mem / 1024
        print(f"  n={n:>6}: {val:>6.1f} {unit}")
    print(f"\nThis justifies LITE_THRESHOLD=10000 for auto-skipping Hat matrix.")
    print("Done.")


if __name__ == "__main__":
    main()
