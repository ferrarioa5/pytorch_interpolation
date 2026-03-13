


import os
import sys

# Fix sys.path so the installed C++/CUDA extension is found
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.dirname(_script_dir)
_saved_path = sys.path.copy()
sys.path = [p for p in sys.path if os.path.abspath(p) not in (_script_dir, _repo_dir)]

from pytorch_interpolation import RegularGridInterpolator as RegularGridInterpolator
from pytorch_interpolation import RegularGridInterpolatorGridSample as GSInterp

sys.path = _saved_path

import torch
import numpy as np
import scipy.interpolate as scipy_interpolate
import time
import matplotlib.pyplot as plt

# Try importing torch_interpolations (optional)
try:
    import torch_interpolations as torch_interpolate
    HAS_TORCH_INTERP = True
except ImportError:
    HAS_TORCH_INTERP = False
    print("[WARN] torch_interpolations not available — skipping those benchmarks")

torch.set_default_dtype(torch.float64)

N1 = 5000
N2 = 5000

def time_cpu_pytorch_interp(N):
    device = "cpu"
    torch.set_num_threads(8)
    x      = torch.linspace(0,1,N1).to(device)
    y      = torch.linspace(0,1,N2).to(device)
    xpt    = torch.rand(N).to(device)
    ypt    = torch.rand(N).to(device)
    X, Y   = torch.meshgrid(x, y, indexing="ij")
    F      = torch.sin(X)*torch.sin(Y)
    interp = RegularGridInterpolator((x,y),F, fill_value=0.0, method=0)
    start  = time.time()
    G      = interp(xpt, ypt)
    return time.time()-start

def time_cuda_pytorch_interp(N):
    """Custom CUDA kernel — float64 (default dtype)."""
    device = "cuda"
    start  = torch.cuda.Event(enable_timing=True)
    end    = torch.cuda.Event(enable_timing=True)
    x      = torch.linspace(0,1,N1).to(device)
    y      = torch.linspace(0,1,N2).to(device)
    xpt    = torch.rand(N).to(device)
    ypt    = torch.rand(N).to(device)
    X, Y   = torch.meshgrid(x, y, indexing="ij")
    F      = torch.sin(X)*torch.sin(Y)
    torch.cuda.synchronize()
    interp      = RegularGridInterpolator((x,y),F, fill_value=0.0, method=0)
    start.record()
    G  = interp(xpt, ypt)
    end.record()
    torch.cuda.synchronize()
    return 1e-3*(start.elapsed_time(end)) # conversion to sec

def time_cuda_pytorch_interp_f32(N):
    """Custom CUDA kernel — float32 (fair comparison with grid_sample)."""
    device = "cuda"
    start  = torch.cuda.Event(enable_timing=True)
    end    = torch.cuda.Event(enable_timing=True)
    x      = torch.linspace(0,1,N1, dtype=torch.float32, device=device)
    y      = torch.linspace(0,1,N2, dtype=torch.float32, device=device)
    xpt    = torch.rand(N, dtype=torch.float32, device=device)
    ypt    = torch.rand(N, dtype=torch.float32, device=device)
    X, Y   = torch.meshgrid(x, y, indexing="ij")
    F      = torch.sin(X)*torch.sin(Y)
    torch.cuda.synchronize()
    interp = RegularGridInterpolator((x,y),F, fill_value=0.0, method=0)
    start.record()
    G  = interp(xpt, ypt)
    end.record()
    torch.cuda.synchronize()
    return 1e-3*(start.elapsed_time(end)) # conversion to sec

def time_cuda_gs_bilinear(N):
    device = "cuda"
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    x      = torch.linspace(0,1,N1, dtype=torch.float64).to(device)
    y      = torch.linspace(0,1,N2, dtype=torch.float64).to(device)
    xpt    = torch.rand(N, dtype=torch.float64).to(device)
    ypt    = torch.rand(N, dtype=torch.float64).to(device)
    X, Y   = torch.meshgrid(x, y, indexing="ij")
    F      = torch.sin(X)*torch.sin(Y)
    torch.cuda.synchronize()
    interp = GSInterp((x, y), F.float(), fill_value=0.0, method=0)
    # grid_sample needs float32 — cast query points
    xpt_f = xpt.float()
    ypt_f = ypt.float()
    start_ev.record()
    G = interp(xpt_f, ypt_f)
    end_ev.record()
    torch.cuda.synchronize()
    return 1e-3*(start_ev.elapsed_time(end_ev))

def time_cpu_gs_bilinear(N):
    device = "cpu"
    torch.set_num_threads(8)
    x      = torch.linspace(0,1,N1, dtype=torch.float32).to(device)
    y      = torch.linspace(0,1,N2, dtype=torch.float32).to(device)
    xpt    = torch.rand(N, dtype=torch.float32).to(device)
    ypt    = torch.rand(N, dtype=torch.float32).to(device)
    X, Y   = torch.meshgrid(x, y, indexing="ij")
    F      = torch.sin(X)*torch.sin(Y)
    interp = GSInterp((x, y), F, fill_value=0.0, method=0)
    start  = time.time()
    G      = interp(xpt, ypt)
    return time.time()-start

def time_scipy_interp(N):
    x      = np.linspace(0,1,N1)
    y      = np.linspace(0,1,N2)
    xpt    = np.random.rand(N)
    ypt    = np.random.rand(N)
    X, Y   = np.meshgrid(x, y, indexing="ij")
    F      = np.sin(X)*np.sin(Y)
    interp = scipy_interpolate.RegularGridInterpolator((x, y), F)
    start  = time.time()
    G_rgi  = interp(np.array([xpt, ypt]).T)
    return time.time()-start

def time_cpu_torch_interpolations(N):
    if not HAS_TORCH_INTERP:
        return None
    device = "cpu"
    torch.set_num_threads(8)
    x      = torch.linspace(0,1,N1).to(device)
    y      = torch.linspace(0,1,N2).to(device)
    xpt    = torch.rand(N).to(device)
    ypt    = torch.rand(N).to(device)
    X, Y   = torch.meshgrid(x, y, indexing="ij")
    F      = torch.sin(X)*torch.sin(Y)
    interp = torch_interpolate.RegularGridInterpolator((x, y), F)
    start  = time.time()
    G_rgi  = interp([xpt, ypt])
    return time.time()-start

def time_cuda_torch_interpolations(N):
    if not HAS_TORCH_INTERP:
        return None
    device = "cuda"
    start  = torch.cuda.Event(enable_timing=True)
    end    = torch.cuda.Event(enable_timing=True)
    x      = torch.linspace(0,1,N1).to(device)
    y      = torch.linspace(0,1,N2).to(device)
    xpt    = torch.rand(N).to(device)
    ypt    = torch.rand(N).to(device)
    X, Y   = torch.meshgrid(x, y, indexing="ij")
    F      = torch.sin(X)*torch.sin(Y)
    interp = torch_interpolate.RegularGridInterpolator((x, y), F)
    torch.cuda.synchronize()
    start.record()
    G_rgi  = interp([xpt, ypt])
    end.record()
    torch.cuda.synchronize()
    return 1e-3*(start.elapsed_time(end)) # conversion to sec


Ns   = 2**torch.arange(4,26)
time_pytorch_cpu = []
time_pytorch_cuda = []
time_pytorch_cuda_f32 = []
time_gs_cuda_bilinear = []
time_gs_cpu_bilinear = []
time_scipy = []
time_torch_interpolations_cpu = []
time_torch_interpolations_gpu = []

for N in Ns:
    print(f"  N = {N:>10d} ...", end=" ", flush=True)
    time_pytorch_cpu.append(time_cpu_pytorch_interp(N))
    time_pytorch_cuda.append(time_cuda_pytorch_interp(N))
    time_pytorch_cuda_f32.append(time_cuda_pytorch_interp_f32(N))
    time_gs_cuda_bilinear.append(time_cuda_gs_bilinear(N))
    time_gs_cpu_bilinear.append(time_cpu_gs_bilinear(N))
    time_scipy.append(time_scipy_interp(N))
    t_ti_cpu = time_cpu_torch_interpolations(N)
    t_ti_gpu = time_cuda_torch_interpolations(N)
    time_torch_interpolations_cpu.append(t_ti_cpu)
    time_torch_interpolations_gpu.append(t_ti_gpu)
    print("done")




# ==== plotting ====
plt.figure(figsize=(8,6))
plot_type = plt.loglog
ms=10
lw=1.5
# Color scheme: same hue for GPU/CPU pairs, solid for GPU, dashed for CPU
plot_type(Ns, time_pytorch_cuda, color='tab:blue', marker='.', ls='-',  label="pytorch_interp GPU (CUDA f64)",    ms=ms, lw=lw)
plot_type(Ns, time_pytorch_cuda_f32, color='tab:cyan', marker='s', ls='-',  label="pytorch_interp GPU (CUDA f32)", ms=ms-2, lw=lw)
plot_type(Ns, time_pytorch_cpu,  color='tab:blue', marker='.', ls='--', label="pytorch_interp CPU (C++/OpenMP)", ms=ms, lw=lw)
plot_type(Ns, time_gs_cuda_bilinear,     color='tab:green', marker='^', ls='-',  label="grid_sample GPU bilinear",     ms=ms-2, lw=lw)
plot_type(Ns, time_gs_cpu_bilinear,      color='tab:green', marker='^', ls='--', label="grid_sample CPU bilinear",     ms=ms-2, lw=lw)
plot_type(Ns, time_scipy, color='tab:red', marker='.', ls='-', label="scipy", ms=ms, lw=lw)
if HAS_TORCH_INTERP:
    plot_type(Ns, time_torch_interpolations_gpu, color='tab:purple', marker='.', ls='-',  label="torch_interpolations GPU", ms=ms, lw=lw)
    plot_type(Ns, time_torch_interpolations_cpu, color='tab:purple', marker='.', ls='--', label="torch_interpolations CPU", ms=ms, lw=lw)
plt.xlabel("Number of interpolated points")
plt.ylabel("Execution time (sec)")
plt.legend(fontsize=8)
plt.title(f"Performance comparison (grid {N1}×{N2})")
plt.xscale("log", base=10)
plt.yscale("log", base=10)
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()

figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../figures/")
plt.savefig(os.path.join(figures_dir, "performance"))




