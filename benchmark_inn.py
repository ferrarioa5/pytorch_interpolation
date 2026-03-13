"""
Benchmark: INN (Interpolating Neural Network) vs Classical Interpolation
=========================================================================

Compares the INN approach (from pyinn / Nature Comms 2025) against the
classical grid-based interpolators already in ``pytorch_interpolation``.

**Classical interpolation** (bilinear, biquadratic, grid_sample):
    Given *exact* function values on a regular grid, interpolate at
    arbitrary query points in O(1) per point — no training needed.

**INN interpolation** (this benchmark):
    Given *scattered* training samples (x, f(x)), learn a CP-decomposed
    piecewise-linear function that approximates f.  Requires a training
    phase but can generalise from noisy / incomplete data.

Metrics compared:
    1. Accuracy — L∞ (max) and L2 (mean) error vs ground truth.
    2. Prediction speed — wall-clock time per batch.
    3. Training cost — total training time for INN.
    4. Parameter count.

Usage:
    python benchmark_inn.py
"""

import os
import sys
import time
import math
import torch
import numpy as np

# ---------------------------------------------------------------------------
#  Imports — resolve path to local package
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))

# Try the installed package first (without the script directory on sys.path),
# then fall back to a direct import from the source tree.
_saved_path = sys.path.copy()
sys.path = [p for p in sys.path if os.path.abspath(p) != _script_dir]

try:
    from pytorch_interpolation.inn import INNInterpolator
    HAS_INN = True
except ImportError:
    HAS_INN = False

try:
    from pytorch_interpolation import RegularGridInterpolatorGridSample as GSInterp
    HAS_GS = True
except ImportError:
    HAS_GS = False

sys.path = _saved_path

# Fallback: import from source tree if the installed package is unavailable
if not HAS_INN:
    try:
        sys.path.insert(0, _script_dir)
        from pytorch_interpolation.inn import INNInterpolator
        HAS_INN = True
    except ImportError:
        pass

if not HAS_GS:
    try:
        sys.path.insert(0, _script_dir)
        from pytorch_interpolation.grid_sample_interp import (
            RegularGridInterpolatorGridSample as GSInterp,
        )
        HAS_GS = True
    except ImportError:
        pass

if not HAS_INN:
    print(
        "[ERROR] INNInterpolator not importable. "
        "Ensure pytorch_interpolation is installed or run from the package directory."
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


# ---------------------------------------------------------------------------
#  Test functions
# ---------------------------------------------------------------------------

def fn_sincos(x, y):
    """f(x,y) = cos(πx) · sin(πy)  — smooth, oscillatory."""
    return torch.cos(x * math.pi) * torch.sin(y * math.pi)


def fn_peaks(x, y):
    """MATLAB 'peaks'-like function — multimodal."""
    return (
        3 * (1 - x) ** 2 * torch.exp(-(x ** 2) - (y + 1) ** 2)
        - 10 * (x / 5 - x ** 3 - y ** 5) * torch.exp(-x ** 2 - y ** 2)
        - 1 / 3 * torch.exp(-(x + 1) ** 2 - y ** 2)
    )


TEST_FUNCTIONS = {
    "cos·sin": (fn_sincos, (-1.0, 1.0), (-1.0, 1.0)),
    "peaks":   (fn_peaks,  (-3.0, 3.0), (-3.0, 3.0)),
}


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def make_grid(fn, nx, ny, x_range, y_range):
    x = torch.linspace(x_range[0], x_range[1], nx, device=DEVICE, dtype=DTYPE)
    y = torch.linspace(y_range[0], y_range[1], ny, device=DEVICE, dtype=DTYPE)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    F = fn(xx, yy)
    return x, y, F


def make_query(n, x_range, y_range):
    xpt = torch.rand(n, device=DEVICE, dtype=DTYPE) * (x_range[1] - x_range[0]) + x_range[0]
    ypt = torch.rand(n, device=DEVICE, dtype=DTYPE) * (y_range[1] - y_range[0]) + y_range[0]
    return xpt, ypt


def time_fn(fn, *args, n_warmup=5, n_iter=50):
    for _ in range(n_warmup):
        fn(*args)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(*args)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter


def header(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
#  Benchmark 1 — Accuracy comparison
# ---------------------------------------------------------------------------
def benchmark_accuracy():
    header("BENCHMARK 1: Accuracy — INN vs Classical Interpolation")
    n_test = 50_000

    for fn_name, (fn, xr, yr) in TEST_FUNCTIONS.items():
        print(f"\n  Function: {fn_name}  (test points: {n_test})")
        print(f"  {'Method':<35s} {'max_err':>10s}  {'mean_err':>10s}  {'RMSE':>10s}")
        print("  " + "-" * 72)

        xpt, ypt = make_query(n_test, xr, yr)
        exact = fn(xpt, ypt)

        # --- Classical: grid_sample bilinear ---
        if HAS_GS:
            for grid_n in [64, 128, 256]:
                x, y, F = make_grid(fn, grid_n, grid_n, xr, yr)
                gs = GSInterp((x, y), F, fill_value="nearest", method=0)
                pred = gs(xpt, ypt)
                err = (pred - exact).abs()
                label = f"GridSample bilinear {grid_n}×{grid_n}"
                print(
                    f"  {label:<35s} {err.max().item():>10.2e}"
                    f"  {err.mean().item():>10.2e}"
                    f"  {err.pow(2).mean().sqrt().item():>10.2e}"
                )

        # --- INN with different configs ---
        for n_modes, n_seg in [(10, 10), (20, 20), (20, 40), (40, 40)]:
            x_train_pts = torch.rand(5000, 2, device=DEVICE, dtype=DTYPE)
            x_train_pts[:, 0] = x_train_pts[:, 0] * (xr[1] - xr[0]) + xr[0]
            x_train_pts[:, 1] = x_train_pts[:, 1] * (yr[1] - yr[0]) + yr[0]
            y_train_pts = fn(x_train_pts[:, 0], x_train_pts[:, 1]).unsqueeze(-1)

            model = INNInterpolator(
                input_dim=2, output_dim=1,
                n_modes=n_modes, n_segments=n_seg,
            ).to(DEVICE)

            model.fit(
                x_train_pts, y_train_pts,
                epochs=2000, lr=1e-3, batch_size=512,
                patience=100, verbose=False,
            )

            x_test_2d = torch.stack([xpt, ypt], dim=-1)
            pred = model.predict(x_test_2d)
            err = (pred - exact).abs()
            n_params = model.num_parameters
            label = f"INN M={n_modes} seg={n_seg} ({n_params} params)"
            print(
                f"  {label:<35s} {err.max().item():>10.2e}"
                f"  {err.mean().item():>10.2e}"
                f"  {err.pow(2).mean().sqrt().item():>10.2e}"
            )


# ---------------------------------------------------------------------------
#  Benchmark 2 — Prediction speed
# ---------------------------------------------------------------------------
def benchmark_speed():
    header("BENCHMARK 2: Prediction Speed (50K queries)")
    fn, xr, yr = fn_sincos, (-1.0, 1.0), (-1.0, 1.0)
    n_pts = 50_000
    xpt, ypt = make_query(n_pts, xr, yr)
    x_test_2d = torch.stack([xpt, ypt], dim=-1)

    print(f"  {'Method':<35s}  {'time (ms)':>10s}")
    print("  " + "-" * 50)

    if HAS_GS:
        for grid_n in [128, 256, 512]:
            x, y, F = make_grid(fn, grid_n, grid_n, xr, yr)
            gs = GSInterp((x, y), F, fill_value="nearest", method=0)
            dt = time_fn(gs, xpt, ypt) * 1000
            label = f"GridSample bilinear {grid_n}×{grid_n}"
            print(f"  {label:<35s}  {dt:>10.3f}")

    for n_modes, n_seg in [(10, 10), (20, 20), (40, 40)]:
        # Quick train
        x_tr = torch.rand(2000, 2, device=DEVICE, dtype=DTYPE)
        x_tr[:, 0] = x_tr[:, 0] * 2 - 1
        x_tr[:, 1] = x_tr[:, 1] * 2 - 1
        y_tr = fn(x_tr[:, 0], x_tr[:, 1]).unsqueeze(-1)

        model = INNInterpolator(2, 1, n_modes, n_seg).to(DEVICE)
        model.fit(x_tr, y_tr, epochs=500, lr=1e-3, verbose=False)

        model.eval()
        with torch.no_grad():
            dt = time_fn(model, x_test_2d) * 1000
        label = f"INN M={n_modes} seg={n_seg}"
        print(f"  {label:<35s}  {dt:>10.3f}")


# ---------------------------------------------------------------------------
#  Benchmark 3 — Training cost
# ---------------------------------------------------------------------------
def benchmark_training():
    header("BENCHMARK 3: INN Training Cost")
    fn, xr, yr = fn_sincos, (-1.0, 1.0), (-1.0, 1.0)

    configs = [
        (1000,  10, 10),
        (2000,  20, 20),
        (5000,  20, 40),
        (5000,  40, 40),
        (10000, 40, 40),
    ]

    print(
        f"  {'n_train':>8s}  {'M':>4s}  {'seg':>4s}  {'params':>8s}"
        f"  {'time (s)':>9s}  {'RMSE':>10s}"
    )
    print("  " + "-" * 60)

    for n_train, n_modes, n_seg in configs:
        x_tr = torch.rand(n_train, 2, device=DEVICE, dtype=DTYPE)
        x_tr[:, 0] = x_tr[:, 0] * 2 - 1
        x_tr[:, 1] = x_tr[:, 1] * 2 - 1
        y_tr = fn(x_tr[:, 0], x_tr[:, 1]).unsqueeze(-1)

        model = INNInterpolator(2, 1, n_modes, n_seg).to(DEVICE)
        info = model.fit(
            x_tr, y_tr,
            epochs=2000, lr=1e-3, batch_size=512,
            patience=100, verbose=False,
        )

        # Test
        xpt, ypt = make_query(50_000, xr, yr)
        exact = fn(xpt, ypt)
        pred = model.predict(torch.stack([xpt, ypt], dim=-1))
        rmse = (pred - exact).pow(2).mean().sqrt().item()

        print(
            f"  {n_train:>8d}  {n_modes:>4d}  {n_seg:>4d}"
            f"  {model.num_parameters:>8d}  {info['train_time']:>9.2f}"
            f"  {rmse:>10.2e}"
        )


# ---------------------------------------------------------------------------
#  Benchmark 4 — Effect of grid resolution (INN segments vs classical grid)
# ---------------------------------------------------------------------------
def benchmark_resolution():
    header("BENCHMARK 4: Resolution Scaling — INN segments vs Grid points")
    fn, xr, yr = fn_sincos, (-1.0, 1.0), (-1.0, 1.0)
    n_test = 50_000
    xpt, ypt = make_query(n_test, xr, yr)
    exact = fn(xpt, ypt)

    resolutions = [16, 32, 64, 128, 256]

    print(f"  {'Res':>5s}  {'GS bilinear RMSE':>18s}  {'INN M=20 RMSE':>18s}  {'INN train time':>15s}")
    print("  " + "-" * 65)

    for res in resolutions:
        # Classical
        gs_rmse_str = "N/A"
        if HAS_GS:
            x, y, F = make_grid(fn, res, res, xr, yr)
            gs = GSInterp((x, y), F, fill_value="nearest", method=0)
            pred_gs = gs(xpt, ypt)
            gs_rmse = (pred_gs - exact).pow(2).mean().sqrt().item()
            gs_rmse_str = f"{gs_rmse:.2e}"

        # INN with same "resolution"
        x_tr = torch.rand(5000, 2, device=DEVICE, dtype=DTYPE)
        x_tr[:, 0] = x_tr[:, 0] * 2 - 1
        x_tr[:, 1] = x_tr[:, 1] * 2 - 1
        y_tr = fn(x_tr[:, 0], x_tr[:, 1]).unsqueeze(-1)

        model = INNInterpolator(2, 1, n_modes=20, n_segments=res).to(DEVICE)
        info = model.fit(
            x_tr, y_tr,
            epochs=2000, lr=1e-3, batch_size=512,
            patience=100, verbose=False,
        )

        pred_inn = model.predict(torch.stack([xpt, ypt], dim=-1))
        inn_rmse = (pred_inn - exact).pow(2).mean().sqrt().item()

        print(
            f"  {res:>5d}  {gs_rmse_str:>18s}"
            f"  {inn_rmse:>18.2e}  {info['train_time']:>14.2f}s"
        )


# ---------------------------------------------------------------------------
#  Benchmark 5 — INN for higher-dimensional data (3-D)
# ---------------------------------------------------------------------------
def benchmark_3d():
    header("BENCHMARK 5: INN in 3-D — sin(x)·cos(y)·exp(−0.1z²)")

    def fn3d(x, y, z):
        return torch.sin(x) * torch.cos(y) * torch.exp(-0.1 * z ** 2)

    xr = yr = zr = (-2.0, 2.0)
    n_test = 50_000

    xpt = torch.rand(n_test, device=DEVICE, dtype=DTYPE) * 4 - 2
    ypt = torch.rand(n_test, device=DEVICE, dtype=DTYPE) * 4 - 2
    zpt = torch.rand(n_test, device=DEVICE, dtype=DTYPE) * 4 - 2
    exact = fn3d(xpt, ypt, zpt)

    configs = [
        (5000,  10, 10),
        (5000,  20, 20),
        (10000, 20, 40),
        (10000, 40, 40),
    ]

    print(
        f"  {'n_train':>8s}  {'M':>4s}  {'seg':>4s}  {'params':>8s}"
        f"  {'time (s)':>9s}  {'RMSE':>10s}  {'max_err':>10s}"
    )
    print("  " + "-" * 68)

    for n_train, n_modes, n_seg in configs:
        x_tr = torch.rand(n_train, 3, device=DEVICE, dtype=DTYPE)
        x_tr = x_tr * 4 - 2  # [-2, 2]
        y_tr = fn3d(x_tr[:, 0], x_tr[:, 1], x_tr[:, 2]).unsqueeze(-1)

        model = INNInterpolator(
            input_dim=3, output_dim=1,
            n_modes=n_modes, n_segments=n_seg,
        ).to(DEVICE)

        info = model.fit(
            x_tr, y_tr,
            epochs=3000, lr=1e-3, batch_size=512,
            patience=100, verbose=False,
        )

        x_test_3d = torch.stack([xpt, ypt, zpt], dim=-1)
        pred = model.predict(x_test_3d)
        err = (pred - exact).abs()

        print(
            f"  {n_train:>8d}  {n_modes:>4d}  {n_seg:>4d}"
            f"  {model.num_parameters:>8d}  {info['train_time']:>9.2f}"
            f"  {err.pow(2).mean().sqrt().item():>10.2e}"
            f"  {err.max().item():>10.2e}"
        )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device : {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name()}")
    print(f"INN available      : {HAS_INN}")
    print(f"grid_sample available: {HAS_GS}")

    benchmark_accuracy()
    benchmark_speed()
    benchmark_training()
    benchmark_resolution()
    benchmark_3d()

    header("DONE — Summary")
    print("""
  Classical interpolation (bilinear / grid_sample):
    ✓ Zero training cost — instant interpolation from exact grid values
    ✓ Extremely fast evaluation (cuDNN-backed)
    ✓ Exact at grid points; error depends on grid density
    ✗ Requires full regular grid of function values

  INN (Interpolating Neural Network):
    ✓ Learns from scattered / noisy data — no regular grid needed
    ✓ Compact model (CP decomposition keeps parameter count low)
    ✓ Differentiable end-to-end; integrates into training pipelines
    ✗ Requires training phase (seconds to minutes)
    ✗ Higher evaluation cost than classical interpolation
    ✗ Accuracy depends on training data quantity and hyperparameters

  Recommendation:
    • Use classical interpolation when exact grid values are available
      and speed is critical.
    • Use INN when learning from data, handling noise, or when a
      compact differentiable model is needed in a training pipeline.
""")


if __name__ == "__main__":
    main()
