#!/usr/bin/env python
"""
KNN Systematic Benchmark — Single Configuration Runner

Runs one (data_size, knn_k) configuration:
  1. Generates spatial data with fixed seed
  2. Initializes dataset (with timing for distance computation)
  3. Trains GNNWR for N epochs
  4. Collects metrics: R², RMSE, times, memory
  5. Outputs JSON

Usage:
    python benchmarks/bench_knn_systematic.py \
        --data_size 10000 --knn_k 200 --gpu 0 \
        --output results/knn_n10000_k200.json

    # knn_k=0 means full distance matrix
    python benchmarks/bench_knn_systematic.py \
        --data_size 10000 --knn_k 0 --gpu 0
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

# Suppress deprecation warnings from GNNWR 0.1.17
warnings.filterwarnings("ignore", message=".*deprecated.*")


# ---------------------------------------------------------------------------
# Data Generation (self-contained, same as generate_100k_data.py)
# ---------------------------------------------------------------------------

def generate_spatial_data(n_samples=10000, seed=42):
    """Generate simulated spatially non-stationary data."""
    np.random.seed(seed)

    lng = np.random.uniform(100, 125, n_samples)
    lat = np.random.uniform(20, 45, n_samples)

    x1 = np.clip(500 + 100 * (lat - 30) + np.random.normal(0, 200, n_samples), 0, 5000)
    x2 = 25 - 0.5 * (lat - 30) + np.random.normal(0, 3, n_samples)
    x3 = np.clip(800 + 30 * (lng - 110) + np.random.normal(0, 100, n_samples), 100, 2000)
    x4 = np.clip(0.3 + 0.01 * x2 + 0.0001 * x3 + np.random.normal(0, 0.1, n_samples), 0, 1)

    city_centers = [(116, 40), (121, 31), (113, 23), (104, 30)]
    x5 = np.zeros(n_samples)
    for cx, cy in city_centers:
        dist = np.sqrt((lng - cx) ** 2 + (lat - cy) ** 2)
        x5 += 1000 * np.exp(-dist ** 2 / 50)
    x5 = np.clip(x5 + np.random.exponential(50, n_samples), 10, 5000)

    x6 = np.clip(0.3 * x5 + np.random.normal(0, 50, n_samples), 0, 2000)

    u = (lng - 112.5) / 12.5
    v = (lat - 32.5) / 12.5

    zone_ne = ((u > 0) & (v > 0)).astype(float)
    zone_nw = ((u <= 0) & (v > 0)).astype(float)
    zone_se = ((u > 0) & (v <= 0)).astype(float)
    zone_sw = ((u <= 0) & (v <= 0)).astype(float)

    beta0 = (zone_ne * (120 + 20 * v + 15 * u) +
             zone_nw * (60 + 10 * v - 20 * u) +
             zone_se * (90 - 15 * v + 25 * u) +
             zone_sw * (50 - 25 * v - 10 * u))
    beta1 = (zone_ne * (0.04 + 0.02 * v) +
             zone_nw * (0.03 + 0.01 * np.sin(3 * np.pi * u)) +
             zone_se * (-0.02 - 0.015 * u) +
             zone_sw * (-0.03 + 0.01 * v))
    beta2 = (-1.5 * v - 0.8 * v ** 2 + 0.5 * u * v +
             0.6 * np.sin(4 * np.pi * u) * np.cos(3 * np.pi * v))
    beta3 = (zone_ne * (-0.02 + 0.01 * u) +
             zone_nw * (0.015 - 0.005 * v) +
             zone_se * (-0.03 - 0.02 * np.tanh(5 * u)) +
             zone_sw * (0.01 + 0.008 * u * v))
    beta4 = -30 + 60 * np.tanh(5 * v) + 20 * np.tanh(3 * u) + 15 * u * v

    dist_jingjinji = np.sqrt((lng - 116) ** 2 + (lat - 39) ** 2)
    dist_yangtze = np.sqrt((lng - 121) ** 2 + (lat - 31) ** 2)
    dist_pearl = np.sqrt((lng - 113) ** 2 + (lat - 23) ** 2)
    dist_chengdu = np.sqrt((lng - 104) ** 2 + (lat - 30) ** 2)
    beta5 = (0.005 +
             0.08 * np.exp(-dist_jingjinji ** 2 / 15) +
             0.06 * np.exp(-dist_yangtze ** 2 / 15) +
             0.05 * np.exp(-dist_pearl ** 2 / 15) +
             0.04 * np.exp(-dist_chengdu ** 2 / 15) +
             0.015 * u ** 2)
    beta6 = (0.06 * u + 0.03 * v - 0.04 * u * v +
             0.02 * np.sin(5 * np.pi * u) +
             zone_ne * 0.03 + zone_se * 0.02 - zone_sw * 0.02)

    y = np.clip(
        beta0 + beta1 * x1 + beta2 * x2 + beta3 * x3 +
        beta4 * x4 + beta5 * x5 + beta6 * x6 +
        np.random.normal(0, 5, n_samples),
        5, 500
    )

    return pd.DataFrame({
        'id': np.arange(n_samples),
        'lng': lng, 'lat': lat,
        'elevation': x1, 'temperature': x2, 'precipitation': x3,
        'ndvi': x4, 'population': x5, 'industry': x6,
        'pm25': y,
    })


# ---------------------------------------------------------------------------
# Training config (from plan)
# ---------------------------------------------------------------------------

BATCH_SIZE_MAP = {1000: 128, 5000: 256, 10000: 256, 50000: 512, 100000: 1024}

TRAIN_DEFAULTS = dict(
    epochs=50,
    optimizer="Adam",
    lr=0.01,
    weight_decay=1e-4,
    test_ratio=0.2,
    valid_ratio=0.1,
    sample_seed=42,
    process_fn="minmax_scale",
)

X_COLUMNS = ['elevation', 'temperature', 'precipitation',
             'ndvi', 'population', 'industry']
Y_COLUMN = ['pm25']
SPATIAL_COLUMNS = ['lng', 'lat']


# ---------------------------------------------------------------------------
# Lite DIAGNOSIS monkey-patch (avoids O(n²) Hat matrix OOM for large n)
# Equivalent to feat/diagnosis-lite branch logic, applied at runtime.
# ---------------------------------------------------------------------------

def _patch_diagnosis_for_large_n():
    """
    Monkey-patch gnnwr.utils.DIAGNOSIS to skip Hat matrix computation
    when n > LITE_THRESHOLD. R², RMSE remain accurate; AIC/F-tests
    return NaN placeholders (only used for display during training).
    """
    import torch
    import gnnwr.utils as _utils

    _OrigDIAGNOSIS = _utils.DIAGNOSIS
    LITE_THRESHOLD = 10000

    class LiteDIAGNOSIS:
        def __init__(self, weight, x_data, y_data, y_pred):
            self._device = torch.device('cuda') if weight.is_cuda else torch.device('cpu')
            self.__n = len(y_data)
            self.__k = len(x_data[0])
            self.__y_data = y_data.clone()
            self.__y_pred = y_pred.clone()
            self.__residual = self.__y_data - self.__y_pred
            self.__ssr = torch.sum((self.__y_pred - self.__y_data) ** 2)

            self._lite = self.__n > LITE_THRESHOLD

            if not self._lite:
                # Small dataset: delegate to original DIAGNOSIS
                self._orig = _OrigDIAGNOSIS(weight, x_data, y_data, y_pred)
            else:
                self._orig = None

            self.f3_dict = None
            self.f3_dict_2 = None

        def R2(self):
            return 1 - torch.sum(self.__residual ** 2) / torch.sum(
                (self.__y_data - torch.mean(self.__y_data)) ** 2)

        def RMSE(self):
            return torch.sqrt(torch.sum(self.__residual ** 2) / self.__n)

        def Adjust_R2(self):
            return 1 - (1 - self.R2()) * (self.__n - 1) / (self.__n - self.__k - 1)

        def AIC(self):
            if self._orig is not None:
                return self._orig.AIC()
            # Return scalar tensor placeholder (used in tqdm/TensorBoard)
            return torch.tensor(float('nan'), device=self._device)

        def AICc(self):
            if self._orig is not None:
                return self._orig.AICc()
            return torch.tensor(float('nan'), device=self._device)

        def F1_Global(self):
            if self._orig is not None:
                return self._orig.F1_Global()
            return torch.tensor(float('nan'), device=self._device)

        def F2_Global(self):
            if self._orig is not None:
                return self._orig.F2_Global()
            return torch.tensor([[float('nan')]], device=self._device)

        def F3_Local(self):
            if self._orig is not None:
                return self._orig.F3_Local()
            return {}, {}

        def hat(self):
            if self._orig is not None:
                return self._orig.hat()
            raise RuntimeError("Hat matrix unavailable in lite mode")

    # Patch both the module and models (which may have already imported it)
    _utils.DIAGNOSIS = LiteDIAGNOSIS
    try:
        import gnnwr.models as _models
        _models.DIAGNOSIS = LiteDIAGNOSIS
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Main benchmark logic
# ---------------------------------------------------------------------------

def run_benchmark(data_size, knn_k, gpu, epochs, output_path):
    """Run a single benchmark configuration and save results as JSON."""
    import torch
    from gnnwr import models, datasets

    # Patch DIAGNOSIS to avoid OOM on large datasets
    _patch_diagnosis_for_large_n()

    use_knn = knn_k > 0
    knn_k_param = knn_k if use_knn else None
    knn_label = knn_k if use_knn else "full"

    print(f"[bench] N={data_size}, knn_k={knn_label}, gpu={gpu}, epochs={epochs}")

    # ----- 1. Generate data -----
    t0 = time.perf_counter()
    data = generate_spatial_data(n_samples=data_size, seed=42)
    data_gen_time = time.perf_counter() - t0
    print(f"  data generation: {data_gen_time:.2f}s")

    # ----- 2. Init dataset (distance computation + normalization) -----
    batch_size = BATCH_SIZE_MAP.get(data_size, 256)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t1 = time.perf_counter()
    train_ds, val_ds, test_ds = datasets.init_dataset(
        data=data,
        test_ratio=TRAIN_DEFAULTS["test_ratio"],
        valid_ratio=TRAIN_DEFAULTS["valid_ratio"],
        x_column=X_COLUMNS,
        y_column=Y_COLUMN,
        spatial_column=SPATIAL_COLUMNS,
        id_column=['id'],
        sample_seed=TRAIN_DEFAULTS["sample_seed"],
        process_fn=TRAIN_DEFAULTS["process_fn"],
        batch_size=batch_size,
        knn_k=knn_k_param,
    )
    dist_init_time = time.perf_counter() - t1
    print(f"  dataset init (distance): {dist_init_time:.2f}s")

    # ----- 3. Distance matrix memory -----
    dist_memory_mb = 0.0
    for ds in [train_ds, val_ds, test_ds]:
        if ds.distances is not None:
            dist_memory_mb += ds.distances.nbytes / (1024 ** 2)
    n_reference = train_ds.distances.shape[-1] if train_ds.distances is not None else 0
    print(f"  distance memory: {dist_memory_mb:.1f} MB  (ref dim={n_reference})")

    # ----- 4. Compute OLS baseline (cheap, for reference) -----
    from sklearn.linear_model import LinearRegression
    ols_model = LinearRegression(fit_intercept=True)
    train_df = train_ds.scaledDataframe
    val_df = val_ds.scaledDataframe
    test_df = test_ds.scaledDataframe
    ols_model.fit(train_df[X_COLUMNS], train_df[Y_COLUMN[0]])

    def _ols_r2_rmse(ols_m, df):
        y_true = df[Y_COLUMN[0]].values
        y_pred = ols_m.predict(df[X_COLUMNS])
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(ss_res / len(y_true))
        return float(r2), float(rmse)

    ols_train_r2, ols_train_rmse = _ols_r2_rmse(ols_model, train_df)
    ols_valid_r2, ols_valid_rmse = _ols_r2_rmse(ols_model, val_df)
    ols_test_r2, ols_test_rmse = _ols_r2_rmse(ols_model, test_df)
    print(f"  OLS baseline: train R²={ols_train_r2:.4f}, "
          f"valid R²={ols_valid_r2:.4f}, test R²={ols_test_r2:.4f}, "
          f"RMSE={ols_test_rmse:.4f}")

    # ----- 5. Create and train GNNWR model -----
    model = models.GNNWR(
        train_ds, val_ds, test_ds,
        use_gpu=torch.cuda.is_available(),
        optimizer=TRAIN_DEFAULTS["optimizer"],
        start_lr=TRAIN_DEFAULTS["lr"],
        optimizer_params={"weight_decay": TRAIN_DEFAULTS["weight_decay"]},
        model_name=f"bench_n{data_size}_k{knn_label}",
        model_save_path=os.path.join("gnnwr_models", "bench_systematic"),
        log_path=os.path.join("gnnwr_logs", "bench_systematic"),
    )

    # Count parameters
    n_params = sum(p.numel() for p in model._model.parameters())
    # Detect hidden layer structure
    hidden_layers = model._dense_layers

    print(f"  model params: {n_params}, hidden layers: {hidden_layers}")

    t2 = time.perf_counter()
    model.run(max_epoch=epochs)
    train_time = time.perf_counter() - t2
    time_per_epoch = train_time / epochs
    print(f"  training: {train_time:.2f}s ({time_per_epoch:.2f}s/epoch)")

    # ----- 6. Evaluate: get train/valid/test R² and RMSE -----
    # model.result() evaluates on all splits using the best saved model.
    # The LiteDIAGNOSIS patch makes this safe for large datasets.
    model.result()
    train_r2 = float(model._trainr2)
    valid_r2 = float(model._validr2)
    test_r2 = float(model._GNNWR__testr2)
    test_rmse = float(model._test_diagnosis.RMSE().data)

    print(f"  GNNWR: train R²={train_r2:.4f}, valid R²={valid_r2:.4f}, "
          f"test R²={test_r2:.4f}, RMSE={test_rmse:.4f}")

    # ----- 7. GPU peak memory -----
    peak_gpu_mb = 0.0
    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"  peak GPU memory: {peak_gpu_mb:.1f} MB")

    # ----- 8. Build result dict -----
    n_train = len(train_ds)
    n_valid = len(val_ds)
    n_test = len(test_ds)

    result = {
        "data_size": data_size,
        "knn_k": knn_label,
        "n_reference": n_reference,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "dist_init_time_s": round(dist_init_time, 3),
        "dist_memory_mb": round(dist_memory_mb, 2),
        "train_time_s": round(train_time, 2),
        "time_per_epoch_s": round(time_per_epoch, 3),
        "train_r2": round(train_r2, 6),
        "valid_r2": round(valid_r2, 6),
        "test_r2": round(test_r2, 6),
        "test_rmse": round(test_rmse, 4),
        "ols_train_r2": round(ols_train_r2, 6),
        "ols_valid_r2": round(ols_valid_r2, 6),
        "ols_test_r2": round(ols_test_r2, 6),
        "ols_test_rmse": round(ols_test_rmse, 4),
        "peak_gpu_mb": round(peak_gpu_mb, 1),
        "n_params": n_params,
        "hidden_layers": hidden_layers,
        "batch_size": batch_size,
        "epochs": epochs,
    }

    # ----- 9. Save JSON -----
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved → {output_path}")

    # ----- 10. Cleanup model files -----
    bench_model_dir = os.path.join("gnnwr_models", "bench_systematic")
    if os.path.isdir(bench_model_dir):
        for fn in os.listdir(bench_model_dir):
            if fn.startswith(f"bench_n{data_size}_k{knn_label}"):
                os.remove(os.path.join(bench_model_dir, fn))

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KNN Systematic Benchmark — single configuration runner")
    parser.add_argument("--data_size", type=int, required=True,
                        help="Number of samples (e.g. 1000, 5000, 10000, 50000, 100000)")
    parser.add_argument("--knn_k", type=int, required=True,
                        help="Number of nearest neighbors (0 = full distance)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (default: 0)")
    parser.add_argument("--epochs", type=int, default=TRAIN_DEFAULTS["epochs"],
                        help="Training epochs (default: 50)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: auto-generated)")
    args = parser.parse_args()

    # Set GPU visibility before CUDA initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.output is None:
        knn_label = args.knn_k if args.knn_k > 0 else "full"
        args.output = os.path.join(
            "benchmarks", "results",
            f"knn_n{args.data_size}_k{knn_label}.json")

    run_benchmark(
        data_size=args.data_size,
        knn_k=args.knn_k,
        gpu=args.gpu,
        epochs=args.epochs,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
