"""
Benchmark: Tensor cache vs per-call torch.tensor() creation.

Compares __getitem__ performance with and without tensor caching
for both baseDataset and predictDataset.

Usage:
    python benchmarks/bench_tensor_cache.py
"""

import time

import numpy as np
import torch


SIZES = [1000, 5000, 10000]
N_CALLS = 10000
DIST_DIM = 200  # typical distance vector size


def bench_no_cache(distances, x_data, n_calls):
    """Simulate old __getitem__: create tensor on each call."""
    indices = np.random.randint(0, len(distances), n_calls)
    t0 = time.perf_counter()
    for idx in indices:
        d = torch.tensor(distances[idx], dtype=torch.float)
        x = torch.tensor(x_data[idx], dtype=torch.float)
    return time.perf_counter() - t0


def bench_cached(distances, x_data, n_calls):
    """Simulate new __getitem__: index pre-cached tensor."""
    d_tensor = torch.from_numpy(distances).float()
    x_tensor = torch.from_numpy(x_data).float()
    indices = np.random.randint(0, len(distances), n_calls)
    t0 = time.perf_counter()
    for idx in indices:
        d = d_tensor[idx]
        x = x_tensor[idx]
    return time.perf_counter() - t0


def main():
    print("=" * 60)
    print("Tensor Cache Benchmark")
    print(f"Measuring {N_CALLS} __getitem__ calls per configuration")
    print("=" * 60)

    print(f"\n| {'N':>6} | {'dist_dim':>8} | {'no_cache (s)':>12} | {'cached (s)':>10} | {'speedup':>8} |")
    print(f"|{'-'*8}|{'-'*10}|{'-'*14}|{'-'*12}|{'-'*10}|")

    for n in SIZES:
        distances = np.random.randn(n, DIST_DIM).astype(np.float32)
        x_data = np.random.randn(n, 5).astype(np.float32)

        t_no = bench_no_cache(distances, x_data, N_CALLS)
        t_cached = bench_cached(distances, x_data, N_CALLS)
        speedup = t_no / t_cached if t_cached > 0 else float('inf')

        print(f"| {n:>6} | {DIST_DIM:>8} | {t_no:>12.4f} | {t_cached:>10.4f} | {speedup:>7.1f}x |")

    print("\nDone.")


if __name__ == "__main__":
    main()
