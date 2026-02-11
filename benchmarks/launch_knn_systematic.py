#!/usr/bin/env python
"""
KNN Systematic Benchmark — Multi-GPU Launcher

Generates 28 (data_size, knn_k) configurations, distributes across GPUs,
runs them via subprocess, merges results, and generates markdown tables.

Usage:
    # Use all 8 GPUs
    python benchmarks/launch_knn_systematic.py --gpus 0,1,2,3,4,5,6,7

    # Single GPU (serial)
    python benchmarks/launch_knn_systematic.py --gpus 0

    # Dry run — show config assignments without executing
    python benchmarks/launch_knn_systematic.py --gpus 0,1,2,3 --dry-run

    # Custom epochs
    python benchmarks/launch_knn_systematic.py --gpus 0,1 --epochs 30
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

EXPERIMENT_MATRIX = [
    # (data_size, [knn_k values])   knn_k=0 means full distance
    (1000,   [0, 50, 100, 200, 500]),
    (5000,   [0, 50, 100, 200, 500, 1000]),
    (10000,  [0, 100, 200, 500, 1000, 2000]),
    (50000,  [0, 100, 200, 500, 1000, 2000]),
    (100000, [100, 200, 500, 1000, 2000]),
]

# Rough time estimates per config (seconds) for load balancing
# Larger data + full distance = much longer
TIME_ESTIMATES = {
    # Calibrated from actual runs on A800-SXM4-80GB
    (1000, 0): 10, (1000, 50): 10, (1000, 100): 10,
    (1000, 200): 10, (1000, 500): 10,
    (5000, 0): 25, (5000, 50): 15, (5000, 100): 15,
    (5000, 200): 15, (5000, 500): 18, (5000, 1000): 20,
    (10000, 0): 60, (10000, 100): 30, (10000, 200): 30,
    (10000, 500): 35, (10000, 1000): 35, (10000, 2000): 40,
    (50000, 0): 3600, (50000, 100): 120, (50000, 200): 150,
    (50000, 500): 180, (50000, 1000): 200, (50000, 2000): 250,
    (100000, 100): 200, (100000, 200): 250, (100000, 500): 300,
    (100000, 1000): 350, (100000, 2000): 400,
}


def build_configs():
    """Build flat list of (data_size, knn_k) configs."""
    configs = []
    for data_size, k_values in EXPERIMENT_MATRIX:
        for knn_k in k_values:
            configs.append((data_size, knn_k))
    return configs


def assign_gpus(configs, gpu_ids):
    """
    Assign configs to GPUs with load balancing (longest-job-first).

    Returns dict: {gpu_id: [(data_size, knn_k), ...]}
    """
    # Sort configs by estimated time (descending) for better balancing
    configs_with_time = [
        (TIME_ESTIMATES.get((n, k), 120), n, k) for n, k in configs
    ]
    configs_with_time.sort(reverse=True)

    # Track total estimated load per GPU
    gpu_load = {g: 0 for g in gpu_ids}
    gpu_assignments = defaultdict(list)

    for est_time, n, k in configs_with_time:
        # Assign to least-loaded GPU
        target_gpu = min(gpu_load, key=gpu_load.get)
        gpu_assignments[target_gpu].append((n, k))
        gpu_load[target_gpu] += est_time

    return dict(gpu_assignments), gpu_load


def output_path_for(data_size, knn_k):
    """Generate the output JSON path for a config."""
    knn_label = knn_k if knn_k > 0 else "full"
    return os.path.join(
        "benchmarks", "results",
        f"knn_n{data_size}_k{knn_label}.json")


def launch_config(data_size, knn_k, gpu, epochs):
    """Launch a single benchmark subprocess. Returns Popen object."""
    cmd = [
        sys.executable, "benchmarks/bench_knn_systematic.py",
        "--data_size", str(data_size),
        "--knn_k", str(knn_k),
        "--gpu", str(gpu),
        "--epochs", str(epochs),
        "--output", output_path_for(data_size, knn_k),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env,
    )
    return proc


def run_gpu_queue(gpu_id, configs, epochs):
    """Run configs sequentially on one GPU. Returns list of (config, success)."""
    results = []
    for i, (n, k) in enumerate(configs):
        knn_label = k if k > 0 else "full"
        print(f"  [GPU {gpu_id}] ({i+1}/{len(configs)}) "
              f"N={n}, k={knn_label}")
        t0 = time.perf_counter()
        proc = launch_config(n, k, gpu_id, epochs)
        stdout, _ = proc.communicate()
        elapsed = time.perf_counter() - t0
        success = proc.returncode == 0
        status = "OK" if success else "FAIL"
        print(f"  [GPU {gpu_id}] ({i+1}/{len(configs)}) "
              f"N={n}, k={knn_label} → {status} ({elapsed:.0f}s)")
        if not success:
            print(f"    stdout: {stdout[-500:]}" if stdout else "    (no output)")
        results.append(((n, k), success))
    return results


def merge_results(configs, output_dir="benchmarks/results"):
    """Load individual JSON results and merge into one file."""
    results = []
    for data_size, knn_k in configs:
        path = output_path_for(data_size, knn_k)
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
    return results


def generate_markdown(results, train_config=None):
    """Generate markdown summary tables from results."""
    if not results:
        return "No results to display."

    lines = []
    lines.append("# KNN Systematic Benchmark Results\n")

    # ---- Experiment configuration section ----
    if train_config:
        lines.append("## Experiment Configuration\n")
        lines.append("### Data Generation\n")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append("| Generator | `generate_spatial_data(n_samples=N, seed=42)` |")
        lines.append("| Spatial range | lng ∈ [100, 125], lat ∈ [20, 45] |")
        lines.append("| X variables | elevation, temperature, precipitation, ndvi, population, industry |")
        lines.append("| Y variable | pm25 (simulated spatial non-stationarity) |")
        lines.append("| Spatial non-stationarity | 4-zone coefficients (NE/NW/SE/SW) with tanh transitions |")
        lines.append("| Noise | σ = 5, y clipped to [5, 500] |")
        lines.append("")
        lines.append("### Dataset Split\n")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| test_ratio | {train_config.get('test_ratio', 0.2)} |")
        lines.append(f"| valid_ratio | {train_config.get('valid_ratio', 0.1)} |")
        lines.append(f"| sample_seed | {train_config.get('sample_seed', 42)} |")
        lines.append(f"| process_fn | {train_config.get('process_fn', 'minmax_scale')} |")
        lines.append("| Reference | train set only (default when Reference=None) |")
        lines.append("")

        # Data split details from first result of each size
        by_size_tmp = defaultdict(list)
        for r in results:
            by_size_tmp[r["data_size"]].append(r)
        if any("n_train" in r for r in results):
            lines.append("| N | Train | Valid | Test | Reference (=Train) |")
            lines.append("|---|-------|-------|------|---------------------|")
            for data_size in sorted(by_size_tmp.keys()):
                r0 = by_size_tmp[data_size][0]
                n_train = r0.get("n_train", "?")
                n_valid = r0.get("n_valid", "?")
                n_test = r0.get("n_test", "?")
                lines.append(f"| {data_size:>7,} | {n_train:>5} | {n_valid:>5} | {n_test:>4} | {n_train:>19} |")
            lines.append("")

        lines.append("### Training Hyperparameters\n")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Epochs | {train_config.get('epochs', 50)} |")
        lines.append(f"| Optimizer | {train_config.get('optimizer', 'Adam')} |")
        lines.append(f"| Learning rate | {train_config.get('lr', 0.01)} |")
        lines.append(f"| Weight decay | {train_config.get('weight_decay', 1e-4)} |")
        lines.append("| LR scheduler | StepLR (default in GNNWR) |")
        lines.append("| Dropout | 0.2 (default) |")
        lines.append("| Batch norm | True (default) |")
        lines.append("| Activation | PReLU(init=0.4) (default) |")
        lines.append("| OLS init | True (output layer fixed to OLS coefficients) |")
        batch_str = ", ".join(f"N={k}: {v}" for k, v in sorted(train_config.get("batch_sizes", {}).items()))
        if batch_str:
            lines.append(f"| Batch size | {batch_str} |")
        lines.append("| SWNN layers | auto: `default_dense_layer(input_dim, n_coef)` |")
        lines.append("| Early stopping | disabled |")
        lines.append("")

    # ---- Group results by data_size ----
    by_size = defaultdict(list)
    for r in results:
        by_size[r["data_size"]].append(r)

    # Collect baselines
    full_r2 = {}
    full_mem = {}
    ols_r2 = {}
    ols_rmse = {}
    for r in results:
        if r["knn_k"] == "full":
            full_r2[r["data_size"]] = r["test_r2"]
            full_mem[r["data_size"]] = r["dist_memory_mb"]
        # OLS is the same for all knn_k within a data_size; take from first
        if r["data_size"] not in ols_r2 and "ols_test_r2" in r:
            ols_r2[r["data_size"]] = r["ols_test_r2"]
            ols_rmse[r["data_size"]] = r["ols_test_rmse"]

    # ---- Per-size detail tables ----
    lines.append("## Results by Data Size\n")
    for data_size in sorted(by_size.keys()):
        rows = sorted(by_size[data_size],
                      key=lambda r: (0 if r["knn_k"] == "full" else 1,
                                     r["knn_k"] if isinstance(r["knn_k"], int) else 0))
        n_ref_full = None
        for r in rows:
            if r["knn_k"] == "full":
                n_ref_full = r["n_reference"]
                break

        lines.append(f"### N = {data_size:,}\n")
        if n_ref_full is not None:
            lines.append(f"Reference set size: {n_ref_full:,}\n")

        lines.append(
            "| Method | SWNN Architecture | Params | Dist Time | Dist Mem | Train Time | "
            "Train R² | Valid R² | Test R² | RMSE | GPU MB |")
        lines.append(
            "|--------|-------------------|--------|-----------|----------|------------|"
            "----------|----------|---------|------|--------|")

        # OLS baseline row
        if data_size in ols_r2:
            r0 = rows[0]
            ols_tr = r0.get("ols_train_r2", ols_r2[data_size])
            ols_vr = r0.get("ols_valid_r2", ols_r2[data_size])
            lines.append(f"| {'OLS':>6} | {'linear (6→1)':>17} | {'7':>6} | "
                         f"{'—':>9} | {'—':>8} | {'—':>10} | "
                         f"{ols_tr:>8.4f} | {ols_vr:>8.4f} | "
                         f"{ols_r2[data_size]:>7.4f} | {ols_rmse[data_size]:>4.2f} | {'—':>6} |")

        for r in rows:
            k_str = str(r["knn_k"])
            # SWNN architecture string
            swnn_layers = r.get("swnn_hidden_layers", [])
            swnn_in = r.get("swnn_input_dim", r.get("n_reference", "?"))
            if swnn_layers:
                arch = f"{swnn_in}→{'→'.join(str(s) for s in swnn_layers)}→7"
            else:
                arch = f"{swnn_in}→...→7"
            dist_t = f"{r['dist_init_time_s']:.1f}s"
            dist_m = f"{r['dist_memory_mb']:.0f}MB"
            train_t = f"{r['train_time_s']:.0f}s"
            train_r2 = f"{r['train_r2']:.4f}"
            valid_r2 = f"{r['valid_r2']:.4f}"
            test_r2_val = f"{r['test_r2']:.4f}"
            rmse = f"{r['test_rmse']:.2f}"
            gpu_mb = f"{r['peak_gpu_mb']:.0f}"
            params = f"{r['n_params']:,}"
            lines.append(f"| {'k=' + k_str:>6} | {arch} | {params:>6} | {dist_t:>9} | {dist_m:>8} | "
                         f"{train_t:>10} | {train_r2:>8} | {valid_r2:>8} | "
                         f"{test_r2_val:>7} | {rmse:>4} | {gpu_mb:>6} |")
        lines.append("")

    # ---- R² comparison: OLS vs KNN vs Full ----
    lines.append("## R² Comparison: OLS vs KNN vs Full\n")
    lines.append("| N | OLS R² | knn_k | KNN R² | Full R² | KNN/OLS | KNN/Full |")
    lines.append("|---|--------|-------|--------|---------|---------|----------|")
    for data_size in sorted(by_size.keys()):
        baseline_ols = ols_r2.get(data_size)
        baseline_full = full_r2.get(data_size)
        for r in sorted(by_size[data_size],
                        key=lambda r: (0 if r["knn_k"] == "full" else 1,
                                       r["knn_k"] if isinstance(r["knn_k"], int) else 0)):
            if r["knn_k"] == "full":
                continue
            knn_over_ols = ""
            if baseline_ols is not None and baseline_ols > 0:
                knn_over_ols = f"{r['test_r2'] / baseline_ols * 100:.1f}%"
            else:
                knn_over_ols = "N/A"
            knn_over_full = ""
            if baseline_full is not None and baseline_full > 0:
                knn_over_full = f"{r['test_r2'] / baseline_full * 100:.1f}%"
            else:
                knn_over_full = "N/A"
            ols_str = f"{baseline_ols:.4f}" if baseline_ols is not None else "N/A"
            full_str = f"{baseline_full:.4f}" if baseline_full is not None else "N/A"
            lines.append(f"| {data_size:>7,} | {ols_str:>6} | {r['knn_k']:>5} | "
                         f"{r['test_r2']:.4f} | {full_str:>7} | "
                         f"{knn_over_ols:>7} | {knn_over_full:>8} |")
    lines.append("")

    # ---- Memory savings summary ----
    lines.append("## Memory Savings vs Full Distance\n")
    lines.append("| N | knn_k | Dist Mem | Full Mem | Savings |")
    lines.append("|---|-------|----------|----------|---------|")
    for data_size in sorted(by_size.keys()):
        baseline_mem = full_mem.get(data_size)
        for r in sorted(by_size[data_size],
                        key=lambda r: (0 if r["knn_k"] == "full" else 1,
                                       r["knn_k"] if isinstance(r["knn_k"], int) else 0)):
            if r["knn_k"] == "full":
                continue
            savings = ""
            if baseline_mem is not None and baseline_mem > 0:
                sav_pct = (1 - r["dist_memory_mb"] / baseline_mem) * 100
                savings = f"{sav_pct:.1f}%"
            else:
                savings = "N/A"
            full_str = f"{baseline_mem:.0f}" if baseline_mem is not None else "N/A"
            lines.append(f"| {data_size:>7,} | {r['knn_k']:>5} | "
                         f"{r['dist_memory_mb']:>7.0f}MB | "
                         f"{full_str:>7}MB | "
                         f"{savings:>7} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KNN Systematic Benchmark — multi-GPU launcher")
    parser.add_argument("--gpus", type=str, required=True,
                        help="Comma-separated GPU indices (e.g. 0,1,2,3)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs per config (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show assignments without running")
    parser.add_argument("--results-dir", type=str,
                        default="benchmarks/results",
                        help="Directory for result JSON files")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip running; just merge existing results")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    configs = build_configs()
    assignments, gpu_load = assign_gpus(configs, gpu_ids)

    print("KNN Systematic Benchmark")
    print(f"  {len(configs)} configs across {len(gpu_ids)} GPUs")
    print(f"  Epochs per config: {args.epochs}")
    print()

    # Show assignments
    for gpu_id in sorted(assignments.keys()):
        cfgs = assignments[gpu_id]
        est = gpu_load[gpu_id]
        print(f"  GPU {gpu_id}: {len(cfgs)} configs, "
              f"~{est//60}m estimated")
        for n, k in cfgs:
            knn_label = k if k > 0 else "full"
            print(f"    N={n:>7,}  k={knn_label}")
    print()

    if args.dry_run:
        print("Dry run — exiting.")
        return

    os.makedirs(args.results_dir, exist_ok=True)

    if not args.merge_only:
        # Launch GPU queues in parallel (one thread per GPU)
        import concurrent.futures

        total_start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(gpu_ids)) as executor:
            futures = {}
            for gpu_id in sorted(assignments.keys()):
                future = executor.submit(
                    run_gpu_queue, gpu_id, assignments[gpu_id], args.epochs)
                futures[future] = gpu_id

            all_results = []
            for future in concurrent.futures.as_completed(futures):
                gpu_id = futures[future]
                try:
                    gpu_results = future.result()
                    all_results.extend(gpu_results)
                except Exception as e:
                    print(f"  [GPU {gpu_id}] ERROR: {e}")

        total_time = time.perf_counter() - total_start
        n_ok = sum(1 for _, ok in all_results if ok)
        n_fail = sum(1 for _, ok in all_results if not ok)
        print(f"\nCompleted: {n_ok} OK, {n_fail} FAILED "
              f"in {total_time:.0f}s ({total_time/60:.1f}m)")
        if n_fail > 0:
            print("Failed configs:")
            for (n, k), ok in all_results:
                if not ok:
                    print(f"  N={n}, k={k if k > 0 else 'full'}")
        print()

    # Merge results
    print("Merging results...")
    results = merge_results(configs, args.results_dir)

    # Get hardware info
    hw_info = "unknown"
    try:
        import torch
        if torch.cuda.is_available():
            hw_info = (f"{torch.cuda.device_count()}x "
                       f"{torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    train_config = {
        "epochs": args.epochs,
        "optimizer": "Adam",
        "lr": 0.01,
        "weight_decay": 1e-4,
        "test_ratio": 0.2,
        "valid_ratio": 0.1,
        "sample_seed": 42,
        "process_fn": "minmax_scale",
        "batch_sizes": {1000: 128, 5000: 256, 10000: 256, 50000: 512, 100000: 1024},
    }

    merged = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "hardware": hw_info,
            "n_gpus_used": len(gpu_ids),
            "n_configs": len(configs),
            "n_results": len(results),
            "train_config": train_config,
            "data_config": {
                "generator": "generate_spatial_data(n_samples=N, seed=42)",
                "x_columns": ["elevation", "temperature", "precipitation",
                              "ndvi", "population", "industry"],
                "y_column": "pm25",
                "spatial_columns": ["lng", "lat"],
                "spatial_range": "lng=[100,125], lat=[20,45]",
                "non_stationarity": "4-zone spatially varying coefficients",
                "noise_std": 5,
            },
        },
        "results": results,
    }

    merged_path = os.path.join(args.results_dir,
                               "knn_systematic_results.json")
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"  Merged JSON → {merged_path}")

    # Generate markdown
    md = generate_markdown(results, train_config=train_config)
    md_path = os.path.join(args.results_dir,
                           "knn_systematic_results.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  Markdown   → {md_path}")

    # Print summary to stdout
    print()
    print(md)


if __name__ == "__main__":
    main()
