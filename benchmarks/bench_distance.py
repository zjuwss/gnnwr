"""
Benchmark: Distance computation backends.

Compares:
  - ManhattanDistance: old numpy broadcasting vs new scipy.cdist
  - BasicDistance: scipy vs torch CPU vs torch GPU (if available)

Usage:
    python benchmarks/bench_distance.py
"""

import time
import tracemalloc

import numpy as np
from scipy.spatial import distance

SIZES = [1000, 5000, 10000]
WARMUP = 1
REPEATS = 3
DIMS = 2  # typical spatial coordinates


def _time_fn(fn, warmup=WARMUP, repeats=REPEATS):
    """Run fn with warmup, return mean elapsed seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.mean(times)


def _mem_fn(fn):
    """Run fn and return peak memory in MB."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024


# ── ManhattanDistance: numpy broadcasting vs scipy.cdist ──

def manhattan_numpy(x, y):
    """Old implementation: numpy broadcasting."""
    x = np.float32(x)
    y = np.float32(y)
    return np.float32(np.sum(np.abs(x[:, np.newaxis, :] - y[np.newaxis, :, :]), axis=2))


def manhattan_scipy(x, y):
    """New implementation: scipy.cdist."""
    x = np.float32(x)
    y = np.float32(y)
    return np.float32(distance.cdist(x, y, 'cityblock'))


# ── BasicDistance: scipy vs torch ──

def basic_scipy(x, y):
    x = np.float32(x)
    y = np.float32(y)
    return distance.cdist(x, y, 'euclidean')


def basic_torch_cpu(x, y):
    import torch
    x_t = torch.from_numpy(np.float32(x))
    y_t = torch.from_numpy(np.float32(y))
    return torch.cdist(x_t.unsqueeze(0), y_t.unsqueeze(0)).squeeze(0).numpy()


def basic_torch_gpu(x, y):
    import torch
    x_t = torch.from_numpy(np.float32(x)).cuda()
    y_t = torch.from_numpy(np.float32(y)).cuda()
    d = torch.cdist(x_t.unsqueeze(0), y_t.unsqueeze(0)).squeeze(0)
    return d.cpu().numpy()


def _check_correctness(ref, test, label, atol=1e-4):
    if not np.allclose(ref, test, atol=atol):
        max_diff = np.max(np.abs(ref - test))
        print(f"  WARNING: {label} max diff = {max_diff:.6f}")
    else:
        print(f"  OK: {label} matches reference (atol={atol})")


def main():
    import torch
    has_gpu = torch.cuda.is_available()

    print("=" * 70)
    print("Distance Computation Benchmark")
    print("=" * 70)

    # ── Manhattan Distance ──
    print("\n## ManhattanDistance: numpy broadcasting vs scipy.cdist\n")
    print(f"| {'N':>6} | {'numpy (s)':>10} | {'scipy (s)':>10} | {'speedup':>8} | {'numpy mem':>10} | {'scipy mem':>10} |")
    print(f"|{'-'*8}|{'-'*12}|{'-'*12}|{'-'*10}|{'-'*12}|{'-'*12}|")

    for n in SIZES:
        x = np.random.randn(n, DIMS).astype(np.float32)
        y = np.random.randn(n, DIMS).astype(np.float32)

        t_np = _time_fn(lambda: manhattan_numpy(x, y))
        t_sp = _time_fn(lambda: manhattan_scipy(x, y))
        m_np = _mem_fn(lambda: manhattan_numpy(x, y))
        m_sp = _mem_fn(lambda: manhattan_scipy(x, y))

        ref = manhattan_numpy(x, y)
        test = manhattan_scipy(x, y)
        _check_correctness(ref, test, f"manhattan n={n}")

        print(f"| {n:>6} | {t_np:>10.4f} | {t_sp:>10.4f} | {t_np/t_sp:>7.1f}x | {m_np:>8.1f} MB | {m_sp:>8.1f} MB |")

    # ── BasicDistance ──
    backends = [("scipy", basic_scipy), ("torch_cpu", basic_torch_cpu)]
    if has_gpu:
        backends.append(("torch_gpu", basic_torch_gpu))

    print(f"\n## BasicDistance: backend comparison\n")
    header = f"| {'N':>6} |"
    for name, _ in backends:
        header += f" {name + ' (s)':>12} |"
    header += f" {'best speedup':>13} |"
    print(header)
    sep = f"|{'-'*8}|"
    for _ in backends:
        sep += f"{'-'*14}|"
    sep += f"{'-'*15}|"
    print(sep)

    for n in SIZES:
        x = np.random.randn(n, DIMS).astype(np.float32)
        y = np.random.randn(n, DIMS).astype(np.float32)

        times = {}
        for name, fn in backends:
            times[name] = _time_fn(lambda fn=fn: fn(x, y))

        # correctness check
        ref = basic_scipy(x, y)
        for name, fn in backends[1:]:
            test = fn(x, y)
            _check_correctness(ref, test, f"basic_{name} n={n}")

        row = f"| {n:>6} |"
        base = times["scipy"]
        best_speedup = 1.0
        for name, _ in backends:
            row += f" {times[name]:>12.4f} |"
            if name != "scipy":
                best_speedup = max(best_speedup, base / times[name])
        row += f" {best_speedup:>12.1f}x |"
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
