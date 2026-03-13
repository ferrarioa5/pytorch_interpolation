
"""
3-D trilinear interpolation example using RegularGridInterpolator3D (custom CUDA kernel).

Demonstrates:
    1. Building a 3-D grid and evaluating a known analytic function.
    2. Interpolating at random query points with the custom CUDA trilinear interpolator.
    3. Comparing accuracy against the analytic solution.
    4. Benchmarking GPU vs CPU performance across varying query counts.
"""


import os
from pytorch_interpolation import RegularGridInterpolator3D as GSInterp3D

import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# ── grid parameters ───────────────────────────────────────────────────
Nx, Ny, Nz = 128, 128, 128       # grid resolution
N_query    = 2**21                # number of random query points
dtype      = torch.float32

x1, x2 = -2.0, 2.0
y1, y2 = -2.0, 2.0
z1, z2 = -2.0, 2.0

# Analytic test function
def analytic_fn(x, y, z):
    return torch.sin(x) * torch.cos(y) * torch.exp(-0.1 * z**2)


# ======================================================================
#  1.  Accuracy test (GPU)
# ======================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

x = torch.linspace(x1, x2, Nx, dtype=dtype, device=device)
y = torch.linspace(y1, y2, Ny, dtype=dtype, device=device)
z = torch.linspace(z1, z2, Nz, dtype=dtype, device=device)

dx, dy, dz = float(x[1]-x[0]), float(y[1]-y[0]), float(z[1]-z[0])

# Keep query points one cell inside the boundary
xmin, xmax = x1 + dx, x2 - dx
ymin, ymax = y1 + dy, y2 - dy
zmin, zmax = z1 + dz, z2 - dz

X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
F = analytic_fn(X, Y, Z)

xpt = (xmax - xmin) * torch.rand(N_query, dtype=dtype, device=device) + xmin
ypt = (ymax - ymin) * torch.rand(N_query, dtype=dtype, device=device) + ymin
zpt = (zmax - zmin) * torch.rand(N_query, dtype=dtype, device=device) + zmin

interp = GSInterp3D((x, y, z), F, fill_value=0.0)
G = interp(xpt, ypt, zpt)

exact = analytic_fn(xpt, ypt, zpt)
err = G - exact

print(f"\n--- Accuracy (grid {Nx}×{Ny}×{Nz}, {N_query} query points) ---")
print(f"Max absolute error : {err.abs().max().item():.3e}")
print(f"L2 error / N       : {err.norm().item() / N_query:.3e}")


# ======================================================================
#  2.  Performance benchmark: GPU vs CPU
# ======================================================================
Ns = 2 ** torch.arange(4, 26)
times_gpu = []
times_cpu = []

print("\n--- Performance benchmark ---")

# GPU ─────────────────────────────────────────────────────────────────
if device == "cuda":
    x_g = torch.linspace(x1, x2, Nx, dtype=dtype, device="cuda")
    y_g = torch.linspace(y1, y2, Ny, dtype=dtype, device="cuda")
    z_g = torch.linspace(z1, z2, Nz, dtype=dtype, device="cuda")
    X_g, Y_g, Z_g = torch.meshgrid(x_g, y_g, z_g, indexing="ij")
    F_g = analytic_fn(X_g, Y_g, Z_g)
    interp_gpu = GSInterp3D((x_g, y_g, z_g), F_g, fill_value=0.0)

    for N in Ns:
        xp = (xmax - xmin) * torch.rand(int(N), dtype=dtype, device="cuda") + xmin
        yp = (ymax - ymin) * torch.rand(int(N), dtype=dtype, device="cuda") + ymin
        zp = (zmax - zmin) * torch.rand(int(N), dtype=dtype, device="cuda") + zmin

        # warm-up
        _ = interp_gpu(xp, yp, zp)
        torch.cuda.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        _ = interp_gpu(xp, yp, zp)
        end_ev.record()
        torch.cuda.synchronize()
        times_gpu.append(1e-3 * start_ev.elapsed_time(end_ev))
        print(f"  GPU  N={int(N):>10d}  {times_gpu[-1]:.4e} s")

# CPU ─────────────────────────────────────────────────────────────────
torch.set_num_threads(8)
x_c = torch.linspace(x1, x2, Nx, dtype=dtype)
y_c = torch.linspace(y1, y2, Ny, dtype=dtype)
z_c = torch.linspace(z1, z2, Nz, dtype=dtype)
X_c, Y_c, Z_c = torch.meshgrid(x_c, y_c, z_c, indexing="ij")
F_c = analytic_fn(X_c, Y_c, Z_c)
interp_cpu = GSInterp3D((x_c, y_c, z_c), F_c, fill_value=0.0)

for N in Ns:
    xp = (xmax - xmin) * torch.rand(int(N), dtype=dtype) + xmin
    yp = (ymax - ymin) * torch.rand(int(N), dtype=dtype) + ymin
    zp = (zmax - zmin) * torch.rand(int(N), dtype=dtype) + zmin
    t0 = time.time()
    _ = interp_cpu(xp, yp, zp)
    times_cpu.append(time.time() - t0)
    print(f"  CPU  N={int(N):>10d}  {times_cpu[-1]:.4e} s")


# ======================================================================
#  3.  Plot
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: performance ────────────────────────────────────────────────
ax = axes[0]
if times_gpu:
    ax.loglog(Ns.numpy(), times_gpu, 'b.-', label="GPU (CUDA 3D kernel)", ms=8, lw=1.5)
ax.loglog(Ns.numpy(), times_cpu, 'b.--', label="CPU (CUDA 3D kernel)", ms=8, lw=1.5)
ax.set_xlabel("Number of query points")
ax.set_ylabel("Time (s)")
ax.set_title(f"3-D trilinear performance ({Nx}×{Ny}×{Nz} grid)")
ax.legend(fontsize=9)
ax.grid(True, which="both", ls="--", alpha=0.3)

# ── Right: error histogram ──────────────────────────────────────────
ax2 = axes[1]
err_np = err.cpu().numpy()
ax2.hist(err_np, bins=100, edgecolor="black", linewidth=0.4)
ax2.set_xlabel("Interpolation error")
ax2.set_ylabel("Count")
ax2.set_title(f"Error distribution (max |e|={np.max(np.abs(err_np)):.2e})")
ax2.axvline(0, color="red", ls="--", lw=0.8)

plt.tight_layout()
figures_dir = os.path.join(_script_dir, "..", "figures")
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(os.path.join(figures_dir, "example_3d.png"), dpi=150)
print(f"\nFigure saved to figures/example_3d.png")
