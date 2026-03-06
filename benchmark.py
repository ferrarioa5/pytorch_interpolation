"""
Benchmark & comparison: pytorch_interpolation (C++/CUDA) vs grid_sample_interp (torch.nn.functional.grid_sample)

Compares:
  1. Speed (GPU & CPU, various grid sizes and query counts)
  2. Accuracy (vs analytic function)
  3. .F swap overhead
  4. Boundary / extrapolation behaviour
  5. GPU memory overhead

Architecture differences:
  ┌──────────────────────────────────────────────────────────────────┐
  │                  pytorch_interpolation (original)               │
  │  • Custom C++/CUDA kernels (interp.cpp / interp.cu)            │
  │  • method=0: bilinear  (4-point weighted average)              │
  │  • method=1: biquadratic Lagrange (9-point, 3×3 stencil)      │
  │  • fill_method: 1=padding, 2=linear_extrap, 3=nearest         │
  │  • In-place write into pre-allocated G tensor                  │
  │  • Stores raw grid axes, computes indices via floor()          │
  │  • OpenMP on CPU, 256-thread CUDA blocks                       │
  └──────────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────────┐
  │               grid_sample_interp (new, lilytorch)               │
  │  • torch.nn.functional.grid_sample  (cuDNN-backed)             │
  │  • method=0: bilinear                                          │
  │  • method=1: bicubic (16-point, 4×4 cubic spline stencil)     │
  │  • fill_value="nearest" → padding_mode='border'                │
  │  • fill_value=None/0   → padding_mode='zeros'                  │
  │  • Allocates output internally                                 │
  │  • Normalises physical coords to [-1, 1] (align_corners=True)  │
  └──────────────────────────────────────────────────────────────────┘

Usage:
    python benchmark.py
"""

import os
import sys
import time
import torch
import numpy as np

# ---------------------------------------------------------------------------
#  Imports — handle the fact that this script lives next to the source
#  pytorch_interpolation/ package, which would shadow the *installed* one.
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
# Temporarily remove the script's directory so the installed C++/CUDA
# extension is found instead of the local source directory.
_saved_path = sys.path.copy()
sys.path = [p for p in sys.path if os.path.abspath(p) != _script_dir]

try:
    from pytorch_interpolation import RegularGridInterpolator as OrigInterp
    from pytorch_interpolation import RegularGridInterpolatorGridSample as GSInterp
    HAS_ORIG = True
    HAS_GS = True
except ImportError:
    HAS_ORIG = False
    HAS_GS = False
    print("[WARN] pytorch_interpolation not importable — skipping benchmarks")
finally:
    sys.path = _saved_path  # restore

if not HAS_ORIG and not HAS_GS:
    print("[ERROR] Neither implementation available. Exiting.")
    sys.exit(1)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float32


def analytic_fn(x, y):
    """Smooth test function: cos(πx) * sin(πy)."""
    return torch.cos(x * 3.14159265) * torch.sin(y * 3.14159265)


def make_grid(nx, ny, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0)):
    x = torch.linspace(x_range[0], x_range[1], nx, device=DEVICE, dtype=DTYPE)
    y = torch.linspace(y_range[0], y_range[1], ny, device=DEVICE, dtype=DTYPE)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    F = analytic_fn(xx, yy)
    return x, y, F


def make_query_points(n_pts, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0)):
    xpt = torch.rand(n_pts, device=DEVICE, dtype=DTYPE) * (x_range[1] - x_range[0]) + x_range[0]
    ypt = torch.rand(n_pts, device=DEVICE, dtype=DTYPE) * (y_range[1] - y_range[0]) + y_range[0]
    return xpt, ypt


def time_interp(interp_obj, xpt, ypt, n_warmup=10, n_iter=100):
    """Time interpolation call with warmup and CUDA sync."""
    for _ in range(n_warmup):
        interp_obj(xpt, ypt)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        interp_obj(xpt, ypt)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter


def time_f_swap(interp_obj, F_new, n_warmup=5, n_iter=50):
    """Time the .F setter (data swap overhead)."""
    for _ in range(n_warmup):
        interp_obj.F = F_new
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        interp_obj.F = F_new
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter


def header(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
#  Benchmark 1: Speed vs query count
# ---------------------------------------------------------------------------
def benchmark_speed_vs_query_count():
    header("BENCHMARK 1: Speed vs Query Count  (grid 512×512)")
    nx, ny = 512, 512
    x, y, F = make_grid(nx, ny)
    query_counts = [1_000, 10_000, 100_000, 500_000, 1_000_000]

    for method, method_name in [(0, "bilinear"), (1, "quadratic/bicubic")]:
        print(f"\n  method={method} ({method_name})")
        cols = f"  {'N pts':>10s}"
        if HAS_GS:
            cols += f"  {'grid_sample (ms)':>18s}"
        if HAS_ORIG:
            cols += f"  {'original (ms)':>15s}"
        if HAS_GS and HAS_ORIG:
            cols += f"  {'speedup':>8s}"
        print(cols)
        print("  " + "-" * 60)

        if HAS_GS:
            gs = GSInterp((x, y), F, fill_value="nearest", method=method)
        if HAS_ORIG:
            orig = OrigInterp((x, y), F, fill_value="nearest", method=method)

        for n_pts in query_counts:
            xpt, ypt = make_query_points(n_pts)
            line = f"  {n_pts:>10d}"
            dt_gs = dt_orig = None
            if HAS_GS:
                dt_gs = time_interp(gs, xpt, ypt) * 1000
                line += f"  {dt_gs:>18.3f}"
            if HAS_ORIG:
                dt_orig = time_interp(orig, xpt, ypt) * 1000
                line += f"  {dt_orig:>15.3f}"
            if HAS_GS and HAS_ORIG and dt_gs and dt_gs > 0:
                line += f"  {dt_orig / dt_gs:>7.2f}x"
            print(line)


# ---------------------------------------------------------------------------
#  Benchmark 2: Speed vs grid size
# ---------------------------------------------------------------------------
def benchmark_speed_vs_grid_size():
    header("BENCHMARK 2: Speed vs Grid Size  (500K query points)")
    n_pts = 500_000
    grid_sizes = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]

    for method, method_name in [(0, "bilinear"), (1, "quadratic/bicubic")]:
        print(f"\n  method={method} ({method_name})")
        cols = f"  {'Grid':>12s}"
        if HAS_GS:
            cols += f"  {'grid_sample (ms)':>18s}"
        if HAS_ORIG:
            cols += f"  {'original (ms)':>15s}"
        if HAS_GS and HAS_ORIG:
            cols += f"  {'speedup':>8s}"
        print(cols)
        print("  " + "-" * 60)

        for nx, ny in grid_sizes:
            x, y, F = make_grid(nx, ny)
            xpt, ypt = make_query_points(n_pts)

            line = f"  {nx:>5d}×{ny:<5d}"
            dt_gs = dt_orig = None
            if HAS_GS:
                gs = GSInterp((x, y), F, fill_value="nearest", method=method)
                dt_gs = time_interp(gs, xpt, ypt) * 1000
                line += f"  {dt_gs:>18.3f}"
            if HAS_ORIG:
                orig = OrigInterp((x, y), F, fill_value="nearest", method=method)
                dt_orig = time_interp(orig, xpt, ypt) * 1000
                line += f"  {dt_orig:>15.3f}"
            if HAS_GS and HAS_ORIG and dt_gs and dt_gs > 0:
                line += f"  {dt_orig / dt_gs:>7.2f}x"
            print(line)


# ---------------------------------------------------------------------------
#  Benchmark 3: Accuracy vs analytic
# ---------------------------------------------------------------------------
def benchmark_accuracy():
    header("BENCHMARK 3: Accuracy vs Analytic Function")
    n_pts = 200_000
    grid_sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]

    for method, method_name in [(0, "bilinear"), (1, "quadratic/bicubic")]:
        print(f"\n  method={method} ({method_name})")
        cols = f"  {'Grid':>12s}"
        if HAS_GS:
            cols += f"  {'gs max_err':>12s}  {'gs mean_err':>12s}"
        if HAS_ORIG:
            cols += f"  {'orig max_err':>13s}  {'orig mean_err':>14s}"
        if HAS_GS and HAS_ORIG:
            cols += f"  {'gs-orig max':>12s}"
        print(cols)
        print("  " + "-" * 90)

        for nx, ny in grid_sizes:
            x, y, F = make_grid(nx, ny)
            # Interior points only to avoid boundary ambiguity
            xpt = torch.rand(n_pts, device=DEVICE, dtype=DTYPE) * 1.8 - 0.9
            ypt = torch.rand(n_pts, device=DEVICE, dtype=DTYPE) * 1.8 - 0.9
            exact = analytic_fn(xpt, ypt)

            line = f"  {nx:>5d}×{ny:<5d}"
            gs_vals = orig_vals = None

            if HAS_GS:
                gs = GSInterp((x, y), F, fill_value="nearest", method=method)
                gs_vals = gs(xpt, ypt)
                gs_max = (gs_vals - exact).abs().max().item()
                gs_mean = (gs_vals - exact).abs().mean().item()
                line += f"  {gs_max:>12.6f}  {gs_mean:>12.6f}"

            if HAS_ORIG:
                orig = OrigInterp((x, y), F, fill_value="nearest", method=method)
                orig_vals = orig(xpt, ypt)
                orig_max = (orig_vals - exact).abs().max().item()
                orig_mean = (orig_vals - exact).abs().mean().item()
                line += f"  {orig_max:>13.6f}  {orig_mean:>14.6f}"

            if HAS_GS and HAS_ORIG and gs_vals is not None and orig_vals is not None:
                diff = (gs_vals - orig_vals).abs().max().item()
                line += f"  {diff:>12.6f}"

            print(line)


# ---------------------------------------------------------------------------
#  Benchmark 4: .F swap overhead
# ---------------------------------------------------------------------------
def benchmark_f_swap():
    header("BENCHMARK 4: .F Swap Overhead  (method=1)")
    grid_sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
    method = 1

    cols = f"  {'Grid':>12s}"
    if HAS_GS:
        cols += f"  {'gs swap (µs)':>14s}"
    if HAS_ORIG:
        cols += f"  {'orig swap (µs)':>16s}"
    print(cols)
    print("  " + "-" * 50)

    for nx, ny in grid_sizes:
        x, y, F = make_grid(nx, ny)
        F_new = analytic_fn(*torch.meshgrid(
            torch.linspace(-1, 1, nx, device=DEVICE, dtype=DTYPE),
            torch.linspace(-1, 1, ny, device=DEVICE, dtype=DTYPE),
            indexing='ij')) * 0.5

        line = f"  {nx:>5d}×{ny:<5d}"
        if HAS_GS:
            gs = GSInterp((x, y), F, fill_value="nearest", method=method)
            dt_gs = time_f_swap(gs, F_new) * 1e6
            line += f"  {dt_gs:>14.1f}"
        if HAS_ORIG:
            orig = OrigInterp((x, y), F, fill_value="nearest", method=method)
            dt_orig = time_f_swap(orig, F_new) * 1e6
            line += f"  {dt_orig:>16.1f}"
        print(line)


# ---------------------------------------------------------------------------
#  Benchmark 5: Boundary / extrapolation behaviour
# ---------------------------------------------------------------------------
def benchmark_boundary():
    header("BENCHMARK 5: Boundary / Extrapolation Behaviour")
    nx, ny = 128, 128
    x, y, F = make_grid(nx, ny)

    xpt_oob = torch.tensor([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], device=DEVICE, dtype=DTYPE)
    ypt_oob = torch.tensor([ 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)

    configs = [
        ("nearest", 0, "bilinear + nearest (border)"),
        ("nearest", 1, "bicubic/quadratic + nearest (border)"),
        (0.0, 0,       "bilinear + zeros"),
        (0.0, 1,       "bicubic/quadratic + zeros"),
    ]

    for fill_value, method, label in configs:
        print(f"\n  Config: {label}")
        cols = f"  {'x':>6s}"
        if HAS_GS:
            cols += f"  {'grid_sample':>12s}"
        if HAS_ORIG:
            cols += f"  {'original':>12s}"
        if HAS_GS and HAS_ORIG:
            cols += f"  {'diff':>10s}"
        print(cols)
        print("  " + "-" * 50)

        gs_vals = orig_vals = None
        if HAS_GS:
            gs = GSInterp((x, y), F, fill_value=fill_value, method=method)
            gs_vals = gs(xpt_oob, ypt_oob)
        if HAS_ORIG:
            orig = OrigInterp((x, y), F, fill_value=fill_value, method=method)
            orig_vals = orig(xpt_oob, ypt_oob)

        for i in range(len(xpt_oob)):
            line = f"  {xpt_oob[i].item():>6.2f}"
            if HAS_GS and gs_vals is not None:
                line += f"  {gs_vals[i].item():>12.6f}"
            if HAS_ORIG and orig_vals is not None:
                line += f"  {orig_vals[i].item():>12.6f}"
            if HAS_GS and HAS_ORIG and gs_vals is not None and orig_vals is not None:
                line += f"  {abs(gs_vals[i].item() - orig_vals[i].item()):>10.6f}"
            print(line)


# ---------------------------------------------------------------------------
#  Benchmark 6: GPU memory overhead
# ---------------------------------------------------------------------------
def benchmark_memory():
    if DEVICE.type != 'cuda':
        print("\n  [SKIP] Memory benchmark requires CUDA")
        return

    header("BENCHMARK 6: GPU Memory Overhead  (method=1)")
    grid_sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
    method = 1

    cols = f"  {'Grid':>12s}  {'F size (KB)':>12s}"
    if HAS_GS:
        cols += f"  {'gs extra (KB)':>15s}"
    if HAS_ORIG:
        cols += f"  {'orig extra (KB)':>17s}"
    print(cols)
    print("  " + "-" * 65)

    for nx, ny in grid_sizes:
        x, y, F = make_grid(nx, ny)
        f_bytes = F.nelement() * F.element_size()
        line = f"  {nx:>5d}×{ny:<5d}  {f_bytes / 1024:>12.1f}"

        if HAS_GS:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
            gs = GSInterp((x, y), F, fill_value="nearest", method=method)
            torch.cuda.synchronize()
            mem_gs = torch.cuda.memory_allocated() - mem_before
            line += f"  {mem_gs / 1024:>15.1f}"
            del gs

        if HAS_ORIG:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
            orig = OrigInterp((x, y), F, fill_value="nearest", method=method)
            torch.cuda.synchronize()
            mem_orig = torch.cuda.memory_allocated() - mem_before
            line += f"  {mem_orig / 1024:>17.1f}"
            del orig

        torch.cuda.empty_cache()
        print(line)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device : {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    if DEVICE.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name()}")
    print(f"Original (pytorch_interpolation) available: {HAS_ORIG}")
    print(f"grid_sample_interp available              : {HAS_GS}")

    benchmark_speed_vs_query_count()
    benchmark_speed_vs_grid_size()
    benchmark_accuracy()
    benchmark_f_swap()
    benchmark_boundary()
    benchmark_memory()

    header("DONE")


if __name__ == "__main__":
    main()
