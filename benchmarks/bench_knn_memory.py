"""
Benchmark: Full distance matrix vs KNN sparse distance memory and time.

Compares memory usage and computation time between:
  - Full distance matrix: O(n*m) memory
  - KNN sparse distance: O(n*k) memory

Usage:
    python benchmarks/bench_knn_memory.py
"""

import time
import tracemalloc

import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

SIZES = [5000, 10000, 20000]
K_VALUES = [50, 100, 200]
DIMS = 2


def full_distance(x, y):
    """Compute full Euclidean distance matrix."""
    return distance.cdist(np.float32(x), np.float32(y), 'euclidean').astype(np.float32)


def knn_distance(x, y, k):
    """Compute KNN sparse distance matrix."""
    x = np.float32(x)
    y = np.float32(y)
    k = min(k, len(y))
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='auto')
    nn.fit(y)
    distances, indices = nn.kneighbors(x)
    return np.float32(distances), indices


def measure_memory(fn):
    """Measure peak memory of fn in MB."""
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024 / 1024


def measure_time(fn):
    """Measure wall time of fn."""
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def main():
    print("=" * 80)
    print("KNN Sparse Distance Memory & Time Benchmark")
    print("=" * 80)

    print(f"\n| {'N':>6} | {'k':>4} | {'full mem (MB)':>13} | {'knn mem (MB)':>12} | {'mem savings':>11} | {'full time (s)':>13} | {'knn time (s)':>12} | {'time ratio':>10} |")
    print(f"|{'-'*8}|{'-'*6}|{'-'*15}|{'-'*14}|{'-'*13}|{'-'*15}|{'-'*14}|{'-'*12}|")

    for n in SIZES:
        # Reference set = same size as query (simulating self-distance)
        x = np.random.randn(n, DIMS).astype(np.float32)
        y = np.random.randn(n, DIMS).astype(np.float32)

        # Full distance matrix: measure once per N
        _, full_mem = measure_memory(lambda: full_distance(x, y))
        _, full_time = measure_time(lambda: full_distance(x, y))

        for k in K_VALUES:
            if k >= n:
                continue

            _, knn_mem = measure_memory(lambda: knn_distance(x, y, k))
            _, knn_time = measure_time(lambda: knn_distance(x, y, k))

            savings = (1 - knn_mem / full_mem) * 100 if full_mem > 0 else 0
            time_ratio = full_time / knn_time if knn_time > 0 else float('inf')

            print(f"| {n:>6} | {k:>4} | {full_mem:>11.1f}  | {knn_mem:>10.1f}  | {savings:>9.1f}%  | {full_time:>11.4f}  | {knn_time:>10.4f}  | {time_ratio:>9.1f}x |")

    # Summary
    print("\nKey insight: KNN sparse distance reduces memory from O(n²) to O(n*k),")
    print(f"enabling datasets that would require {SIZES[-1]**2 * 4 / 1024**3:.1f} GB with full matrices.")
    print("Done.")


if __name__ == "__main__":
    main()
